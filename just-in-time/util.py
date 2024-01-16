import argparse
import random
import numpy as np
import torch
from transformers import (RobertaModel, RobertaTokenizer, RobertaConfig, T5ForConditionalGeneration, T5Config,
                          PLBartTokenizer, PLBartForConditionalGeneration, PLBartConfig)


def parse_jit_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", nargs=2, type=str)
    parser.add_argument("--eval_data_file", nargs=2, type=str)
    parser.add_argument("--test_data_file", nargs=2, type=str)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--pretrained_model", type=str, default="plbart")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')

    parser.add_argument("--manual_feature_size", type=int, default=14)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh")

    parser.add_argument("--max_msg_length", type=int, default=64)
    parser.add_argument("--max_input_tokens", type=int, default=512)

    parser.add_argument("--available_gpu", type=list, default=[1, 2, 3, 0])
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--use_lora", action='store_true')

    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def build_model_tokenizer_config(args):
    model_classes = {
        "codebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/codebert-base"),
        "graphcodebert": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/graphcodebert-base"),
        "codet5": (T5ForConditionalGeneration, RobertaTokenizer, T5Config, "Salesforce/codet5-base"),
        "unixcoder": (RobertaModel, RobertaTokenizer, RobertaConfig, "microsoft/unixcoder-base"),
        "plbart": (PLBartForConditionalGeneration, PLBartTokenizer, PLBartConfig, "uclanlp/plbart-base"),
        "plbart-large": (PLBartForConditionalGeneration, PLBartTokenizer, PLBartConfig, "uclanlp/plbart-large")
    }

    model_class, tokenizer_class, config_class, actual_name = model_classes[args.pretrained_model]

    # load config.
    config = config_class.from_pretrained(actual_name)
    if args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
        config.hidden_size = args.hidden_size
    elif args.pretrained_model in ["codet5", "plbart", "plbart-large"]:
        config.d_model = args.hidden_size
    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.dropout
    # load tokenizer.
    tokenizer = tokenizer_class.from_pretrained(actual_name)
    special_tokens_dict = {"additional_special_tokens": ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    # load pretrained model.
    model = model_class.from_pretrained(actual_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, config


