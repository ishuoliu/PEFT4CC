import os
import dill
import random
import torch
import multiprocessing
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer
import sys
sys.path.append("..")
from util import parse_cmg_args, build_model_tokenizer_config, set_seed
from models.SingleModel import SingleModel


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 manual_feature,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.manual_feature = manual_feature
        self.url = url
        self.task = task
        self.sub_task = sub_task


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 manual_feature,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.manual_feature = manual_feature
        self.url = url


def normalize_df(df, feature_columns):
    df["fix"] = df["fix"].apply(lambda x: float(bool(x)))
    df = df.astype({i: "float32" for i in feature_columns})
    return df[["commit_hash"] + feature_columns]


def read_file_examples(ccfile, fefile, data_num):
    ccdata = pd.read_pickle(ccfile)
    fedata = pd.read_pickle(fefile)

    # store parsed data.
    examples = []

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = preprocessing.scale(fedata[manual_features_columns].to_numpy())
    fedata[manual_features_columns] = manual_features

    commit_ids, labels, msgs, codes = ccdata
    for idx, (commit_id, label, msg, code) in enumerate(zip(commit_ids, labels, msgs, codes)):
        manual_features = fedata[fedata["commit_hash"] == commit_id][manual_features_columns].to_numpy().squeeze()

        added_codes = [' '.join(line.split()) for line in code['added_code']]
        acodes = '<add>'.join([line for line in added_codes if len(line)])
        removed_codes = [' '.join(line.split()) for line in code['removed_code']]
        recodes = '<del>'.join([line for line in removed_codes if len(line)])
        diff = '<add>' + acodes + '<del>' + recodes
        examples.append(
            Example(
                idx=idx,
                source=diff,
                target=msg,
                manual_feature=manual_features,
            )
        )
        if idx + 1 == data_num:
            break

    return examples


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    source_str = example.source
    # print(source_str)

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_input_tokens, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    target_str = example.target
    target_str = target_str.replace('</s>', '<unk>')
    target_ids = tokenizer.encode(target_str, max_length=args.max_output_tokens, padding='max_length',
                                  truncation=True)
    assert target_ids.count(tokenizer.eos_token_id) == 1

    manual_feature = example.manual_feature

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        manual_feature,
        url=example.url
    )


def load_jitfine_data(args, tokenizer, mode, pool):
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    examples = read_file_examples(codechange_file, feature_file, args.data_num)

    tuple_examples = [(example, idx, tokenizer, args, mode) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))

    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    source_mask = all_source_ids.ne(tokenizer.pad_token_id)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    target_mask = all_target_ids.ne(tokenizer.pad_token_id)

    manual_features = torch.tensor([f.manual_feature for f in features])

    data = TensorDataset(all_source_ids, source_mask, all_target_ids, target_mask, manual_features)

    return examples, data


if __name__ == "__main__":
    args = parse_cmg_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device
    torch.cuda.set_device(args.available_gpu[0])
    cpu_cont = multiprocessing.cpu_count()
    args.cpu_cont = cpu_cont
    print(args.cpu_cont)
    pool = multiprocessing.Pool(args.cpu_cont)

    model, tokenizer, config = build_model_tokenizer_config(args)
    mymodel = SingleModel(model, config, tokenizer, args).to(device)
    if args.n_gpu > 1:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=args.available_gpu)

    examples, test_dataset = load_jitfine_data(args, tokenizer, "test", pool)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4)

    for batch in test_dataloader:
        diff_ids, diff_mask, msg_ids, msg_mask, manual_feature = [x.to(args.device) for x in batch]
        print(diff_ids.shape)
        print(diff_mask.shape)
        print(msg_ids.shape)
        print(msg_mask.shape)
        print(manual_feature.shape)

        loss = mymodel(diff_ids, diff_mask, msg_ids, msg_mask)
        print(loss)

        break