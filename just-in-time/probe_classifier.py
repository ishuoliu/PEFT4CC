import torch
import os
import collections
import json
import sys
import csv
import seaborn as sns
import numpy as np
from sklearn import metrics
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from probe_model import MLP
from util import build_model_tokenizer_config, parse_jit_args
from models.SingleModel import SingleModel


def load_data(labels_file, features_path, eval_layer):
    with open(labels_file, 'r', encoding='utf-8', errors='ignore') as labl_reader:
        write_file = features_path + f"/{args.train_type}_{args.task_code}_{args.data_size}.json"
        with open(write_file, 'r') as feat_reader:

            feat_dim = -1
            cat2id = {}
            id2cat = {}

            train_X = []
            train_y = []
            valid_X = []
            valid_y = []
            test_X = []
            test_y = []

            while True:
                label_line = labl_reader.readline()
                if not label_line:
                    break

                feat_jsonl = feat_reader.readline()
                feat_jsonl = json.loads(feat_jsonl)

                # # ********************************************************
                # # **************** FEATURE TOKENS CHECK ******************
                # # ********************************************************
                # print("Length of tokens", len(feat_jsonl['features']))
                # for ix_ in range(len(feat_jsonl['features'])):
                #     print((feat_jsonl['features'][ix_]['token']), end=' ')
                # # ********************************************************

                # *****************
                # average the layer['values'] of every token
                # token_index 0 => layer['values'] => torch.Size([768])
                # token_index 1 => layer['values'] => torch.Size([768])
                # add each item in torch.Size([768]) across all token indices and divide by number of token indices
                # *****************

                all_X = []
                for token_index in range(len(feat_jsonl['features'])):
                    for layer in feat_jsonl['features'][token_index]['layers']:
                        if layer['index'] == eval_layer:
                            all_X.append(layer['values'])

                X = [float(sum(col))/len(col) for col in zip(*all_X)]

                # *****************

                assert(X is not None)
                if feat_dim < 0:
                    feat_dim = len(X)

                split, label, text = label_line.split('\t',2)
                if label not in cat2id:
                    cat2id[label] = len(id2cat)
                    id2cat[cat2id[label]] = label
                y = cat2id[label]

                if  split  == 'tr':
                    train_X.append(X)
                    train_y.append(y)
                elif split == 'va':
                    valid_X.append(X)
                    valid_y.append(y)
                elif split == 'te':
                    test_X.append(X)
                    test_y.append(y)


    train_X = np.array(train_X, dtype=np.float32)
    valid_X = np.array(valid_X, dtype=np.float32)
    test_X  = np.array(test_X, dtype=np.float32)

    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    test_y  = np.array(test_y)
    # print(train_X.shape)

    #print('loaded %d/%d/%d samples; %d labels;'%(train_X.shape[0], valid_X.shape[0], test_X.shape[0], len(cat2id)))
    return train_X, train_y, valid_X, valid_y, test_X, test_y, feat_dim, len(cat2id), cat2id, id2cat


def classify_and_predict(train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes, nhid, id2cat, dropout, seed, features_path):
    classifier_config = {'nhid': nhid, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 20, 'dropout': dropout}
    regs = [10**t for t in range(-5, -1)]
    props, scores = [], []

    # hyper-parameter optimization
    for reg in regs:
        clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=reg, seed=seed, cudaEfficient=True)
        clf.fit(train_X, train_y, validation_data=(dev_X, dev_y))
        scores.append(round(100*clf.score(dev_X, dev_y), 2))
        props.append([reg])
    opt_prop = props[np.argmax(scores)]
    dev_acc = np.max(scores)

    # training
    classifier_config = {'nhid': nhid, 'optim': 'adam', 'batch_size': 1, 'tenacity': 5, 'epoch_size': 20, 'dropout': dropout}
    clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=opt_prop[0], seed=seed, cudaEfficient=True)
    clf.fit(train_X, train_y, validation_data=(dev_X, dev_y))

    # testing
    test_acc = round(100*clf.score(test_X, test_y), 2)

    # to get predictions use id2cat[]
    predictions = clf.predict(test_X)

    # writing orig and pred values to csv
    orig = [int(item)    for item in test_y.tolist()]
    pred = [int(item[0]) for item in predictions.tolist()]

    orig = [id2cat[item] for item in orig]
    pred = [id2cat[item] for item in pred]

    orig_pred = (zip(orig, pred))
    # orig = [int(item[0]) if len(item) > 1 else int(item) for item in orig]
    # pred = [int(item[0]) if len(item) > 1 else int(item) for item in pred]
    # test_acc = metrics.r2_score(orig, pred)

    outpatx = features_path + f"/{args.train_type}_{args.task_code}_{args.data_size}.csv"
    outpath = Path(outpatx)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # with open(outpath, 'w+') as wf:
    #     csv_writer = csv.writer(wf)
    #     csv_writer.writerow(['orig', 'pred'])
    #
    #     for orig_item, pred_item in orig_pred:
    #         csv_writer.writerow([orig_item, pred_item])

    # confusion matrix
    plt.figure()
    labels = list(id2cat.values())
    labels.sort()

    cf_matrix = metrics.confusion_matrix(orig, pred, labels=labels)
    ax = sns.heatmap(cf_matrix, cmap='RdYlGn', annot=True, yticklabels=labels, fmt='g', square=2, linecolor="white")

    ax.figure.subplots_adjust(left = 0.3)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('PRED Labels')
    ax.set_ylabel('ORIG Labels')

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.figure.savefig(outpatx[:-4]+'_CFMX.png')
    plt.close()

    print(test_acc, end='\t')
    return test_acc


def embed_classifier(args):
    if args.task_code == "MAT":
        text_dataset = f"../datasets/probing/fira_diff_nl_{args.data_size}.txt"
    elif args.task_code == "MSK":
        text_dataset = f"../datasets/probing/fira_msk_{args.data_size}.txt"
    elif args.task_code == "MSK_2":
        text_dataset = f"../datasets/probing/fira_len_{args.data_size}.txt"
    else:
        text_dataset = f"../datasets/probing/datasets_{args.task_code}/{args.task_code}_ORIG_{args.data_size}.txt"
    features_path = f"./probing_records/datasets_{args.task_code}/{args.pretrained_model}"
    dropout = 0.0
    seed = 42
    train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes, cat2id, id2cat = load_data(text_dataset, features_path, args.eval_layer)
    test_acc = classify_and_predict(train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes, args.nhid, id2cat, dropout, seed, features_path)


if __name__ == "__main__":
    args = parse_jit_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device


    args.task_code = "MSK_2"
    args.data_size = "1k"     # "100", "1k", "10k"
    for type in ["ft", "lora", "adapter"]:
        args.train_type = type     # "ft", "lora", "adapter"
        args.pretrained_model = "codet5"

        for i in range(13):
            args.eval_layer = i     # 12, 6 for plbart
            args.nhid = 0

            embed_classifier(args)
        print('\n')