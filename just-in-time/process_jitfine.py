import os
import dill
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing
from transformers import RobertaTokenizer

from util import parse_jit_args, build_model_tokenizer_config, set_seed


def normalize_df(df, feature_columns):
    df["fix"] = df["fix"].apply(lambda x: float(bool(x)))
    df = df.astype({i: "float32" for i in feature_columns})
    return df[["commit_hash"] + feature_columns]


def parse_data_file(args, mode):
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)

    # store parsed data.
    examples = []

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = preprocessing.scale(fedata[manual_features_columns].to_numpy())
    fedata[manual_features_columns] = manual_features

    # parse ccdata.
    commit_ids, labels, msgs, codes = ccdata
    for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
        manual_features = fedata[fedata["commit_hash"] == commit_id][manual_features_columns].to_numpy().squeeze()
        examples.append((commit_id, label, msg, code, manual_features))

    if mode == "train":
        random.seed(args.seed)
        random.shuffle(examples)

    return examples


def further_parse(example, tokenizer, args):
    commit_id, label, msg, code, manual_features = example
    label = int(label)
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    added_codes = [' '.join(line.split()) for line in code['added_code']]
    codes = '[ADD]'.join([line for line in added_codes if len(line)])
    added_tokens.extend(tokenizer.tokenize(codes))

    removed_codes = [' '.join(line.split()) for line in code['removed_code']]
    codes = '[DEL]'.join([line for line in removed_codes if len(line)])
    removed_tokens.extend(tokenizer.tokenize(codes))

    input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
    input_tokens = input_tokens[:args.max_input_tokens - 2]
    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label


class JITFineDataset(Dataset):
    def __init__(self, tokenizer, args, mode):
        self.mid_examples = parse_data_file(args, mode)
        self.examples = [further_parse(item, tokenizer, args) for item in self.mid_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


if __name__ == "__main__":
    args = parse_jit_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    examples = parse_data_file(args, "test")
    print(examples[0])

    # train_dataset = JITFineDataset(tokenizer, args, "train")
    # eval_dataset = JITFineDataset(tokenizer, args, "eval")
    # test_dataset = JITFineDataset(tokenizer, args, "test")
    # print(len(train_dataset))
    # print(len(eval_dataset))
    # print(len(test_dataset))


