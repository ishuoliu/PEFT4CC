import torch
import os
import collections
import json
from tqdm import tqdm
from opendelta import AdapterModel
from peft import LoraConfig, TaskType, get_peft_model

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from util import build_model_tokenizer_config, parse_jit_args
from models.ConcatModel import ConcatModel



class InputExample(object):
    def __init__(self, text, unique_id):
        self.text = text
        self.unique_id = unique_id


class InputFeatures(object):
    def __init__(self, tokens, unique_id, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.unique_id = unique_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(text_file):
    examples = []
    unique_id = 0

    with open(text_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break

            text = line.strip().split('\t')[-1]
            examples.append(InputExample(text=text, unique_id=unique_id))
            unique_id += 1
    return examples


def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        cand_tokens = tokenizer.tokenize(example.text)
        if len(cand_tokens) > seq_length - 2:
            ## Account for [CLS] and [SEP] with "- 2"
            cand_tokens = cand_tokens[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(tokens=tokens, unique_id=example.unique_id, input_ids=input_ids, input_mask=input_mask,
                          input_type_ids=input_type_ids))
    return features


def save_features(model, tokenizer, text_dataset, args):
    print(model)
    # convert data to ids
    examples = read_examples(text_dataset)
    features = convert_examples_to_features(examples=examples, seq_length=args.max_input_tokens, tokenizer=tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_indices = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_example_indices)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)

    write_file = args.features_path + f"/{args.train_type}_{args.task_code}_{args.data_size}.json"

    pbar = tqdm(total=len(examples) // args.batch_size)
    with open(write_file, "w") as writer:
        with torch.no_grad():
            for input_ids, input_mask, example_indices in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                if args.pretrained_model in ["plbart", "plbart-large"]:
                    if args.train_type == "lora":
                        outputs = model.encoder.base_model.model.model.encoder(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True)
                    else:
                        outputs = model.encoder.model.encoder(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True)
                    enc_layers = outputs.hidden_states

                    # all_outputs = model.basemodel(input_ids=input_ids, decoder_input_ids=input_ids, output_hidden_states=True)
                    # enc_layers = all_outputs.encoder_hidden_states
                elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
                    if args.train_type == "lora":
                        outputs = model.encoder.base_model.model(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True)
                    elif args.train_type == "raw":
                        outputs = model(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True)
                    else:
                        outputs = model.encoder(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True)
                    enc_layers = outputs.hidden_states

                    # all_outputs = model.basemodel(input_ids=input_ids, decoder_input_ids=input_ids)
                    # enc_layers = all_outputs.encoder_hidden_states
                elif args.pretrained_model in ["codet5"]:
                    outputs = model.encoder.encoder(input_ids=input_ids, attention_mask=input_mask, output_hidden_states=True)
                    enc_layers = outputs.hidden_states

                # print(len(enc_layers))

                for iter_index, example_index in enumerate(example_indices):
                    feature = features[example_index.item()]
                    unique_id = int(feature.unique_id)

                    all_output_features = []
                    for (token_index, token) in enumerate(feature.tokens):
                        all_layers = []     # for one token, contain all layers
                        for layer_index in range(len(enc_layers)):
                            layer_output = enc_layers[int(layer_index)]
                            layer_feat_output = layer_output[iter_index]

                            layers = collections.OrderedDict()
                            layers["index"] = layer_index
                            layers["values"] = [round(hidden_unit.item(), 6) for hidden_unit in layer_feat_output[token_index]]
                            all_layers.append(layers)

                        out_features = collections.OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_output_features.append(out_features)
                        break  # if breaking only [CLS] token will be considered for classification

                    output_json = collections.OrderedDict()
                    output_json["linex_index"] = unique_id
                    output_json["features"] = all_output_features
                    writer.write(json.dumps(output_json) + "\n")

                pbar.update(1)
    pbar.close()
    print('written features to %s' % (write_file))


def embed_extractor(args):
    if args.task_code == "MAT":
        text_dataset = f"../datasets/probing/fira_diff_nl_{args.data_size}.txt"
    elif args.task_code == "MSK":
        text_dataset = f"../datasets/probing/fira_msk_{args.data_size}.txt"
    elif args.task_code == "MSK_2":
        text_dataset = f"../datasets/probing/fira_len_{args.data_size}.txt"
    else:
        text_dataset = f"../datasets/probing/datasets_{args.task_code}/{args.task_code}_ORIG_{args.data_size}.txt"
    features_path = f"./probing_records/datasets_{args.task_code}/{args.pretrained_model}"

    if not os.path.exists(features_path):
        os.makedirs(features_path)

    args.features_path = features_path

    model, tokenizer, config = build_model_tokenizer_config(args)

    if args.train_type == "ft":
        pass
    elif args.train_type == "adapter":
        if args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
            delta_model = AdapterModel(backbone_model=model,
                                       modified_modules=['attention', '[r](\d)+\.output'],
                                       bottleneck_dim=128)
            delta_model.freeze_module(exclude=["deltas", "LayerNorm"], set_state_dict=False)
        elif args.pretrained_model in ["codet5"]:
            delta_model = AdapterModel(backbone_model=model,
                                       modified_modules=['layer.0', 'layer.2', '[r]encoder\.block\.(\d)+\.layer\.[01]'],
                                       bottleneck_dim=128)
            delta_model.freeze_module(exclude=["deltas", "layer_norm"], set_state_dict=False)
        elif args.pretrained_model in ["plbart", "plbart-large"]:
            delta_model = AdapterModel(backbone_model=model,
                                       modified_modules=['self_attn_layer_norm', 'final_layer_norm'],
                                       bottleneck_dim=128)
            delta_model.freeze_module(exclude=["deltas", "self_attn_layer_norm", "final_layer_norm"], set_state_dict=False)

    elif args.train_type == "lora":
        if args.pretrained_model in ["codet5"]:
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=64, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["q", "v"])
        elif args.pretrained_model in ["plbart"]:
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=64, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
        elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32,
                                     lora_dropout=0.1, target_modules=["query", "value"])
        model = get_peft_model(model, peft_config)

    if args.train_type != "raw":
        mymodel = ConcatModel(model, config, tokenizer, args).to(device)

    else:
        mymodel = model.to(device)

    if args.train_type != "raw":
        checkpoint = torch.load(args.save_model_path)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
    save_features(mymodel, tokenizer, text_dataset, args)


if __name__ == "__main__":
    args = parse_jit_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.task_code = "MAT"
    args.data_size = "1k"     # "100", "1k", "10k"
    args.train_type = "lora"     # "ft", "lora", "adapter"
    args.pretrained_model = "codet5"
    args.save_model_path = f"../results/jitfine_lora/{args.pretrained_model}/concat/checkpoints_lr1e-4/" \
                           + "checkpoint-best-f1/model.bin"

    embed_extractor(args)
