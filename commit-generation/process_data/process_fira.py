import json
import re
import torch
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import sys
sys.path.append("..")
from util import parse_cmg_args, build_model_tokenizer_config
from models.SingleModel import SingleModel


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 manual_feature=None,
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
                 manual_feature=None,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.manual_feature = manual_feature
        self.url = url


def read_file_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            diff = js['diff']
            nl = js['nl']

            difflines = diff.split("\n")
            regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
            matchres = re.match(regex, difflines[0])
            if matchres:
                difflines = difflines[1:]  # remove start @@

            difflines = [line for line in difflines if len(line.strip()) > 0]
            map_dic = {"-": 0, "+": 1, " ": 2}

            def f(s):
                if s in map_dic:
                    return map_dic[s]
                else:
                    return 2

            labels = [f(line[0]) for line in difflines]
            difflines = [line[1:].strip() for line in difflines]
            inputstr = ""
            for label, line in zip(labels, difflines):
                if label == 1:
                    inputstr += "<add>" + line
                elif label == 0:
                    inputstr += "<del>" + line
                else:
                    inputstr += "<keep>" + line

            examples.append(
                Example(
                    idx=idx,
                    source=inputstr,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_input_tokens, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    target_str = example.target
    target_str = target_str.replace('</s>', '<unk>')
    target_ids = tokenizer.encode(target_str, max_length=args.max_output_tokens, padding='max_length',
                                  truncation=True)
    assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def load_fira_data(args, tokenizer, mode, pool):
    """ only_src: control whether to return only source ids for bleu evaluating (dev/test) """

    filename = ""
    if mode == "train":
        filename, _ = args.train_data_file
    elif mode == "eval":
        filename, _ = args.eval_data_file
    elif mode == "test":
        filename, _ = args.test_data_file
    examples = read_file_examples(filename, args.data_num)

    tuple_examples = [(example, idx, tokenizer, args, mode) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))

    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    source_mask = all_source_ids.ne(tokenizer.pad_token_id)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    target_mask = all_target_ids.ne(tokenizer.pad_token_id)

    item_num = target_mask.shape[0]
    manual_features = torch.zeros(item_num, 1)

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

    examples, test_dataset = load_fira_data(args, tokenizer, "test", pool)

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