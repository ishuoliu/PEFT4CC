import torch
import torch.nn as nn
from models.Seq2Seq_uni import Seq2Seq_uni
from models.Seq2Seq_bert import Seq2Seq_bert


class SingleModel(nn.Module):
    def __init__(self, basemodel, config, tokenizer, args):
        super(SingleModel, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        if args.pretrained_model in ["codet5", "plbart", "plbart-large"]:
            self.basemodel = basemodel
        elif args.pretrained_model in ["unixcoder"]:
            self.model = Seq2Seq_uni(encoder=basemodel, decoder=basemodel, config=config, args = args,
                                     beam_size=args.beam_size, max_length=args.max_output_tokens,
                                     sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)
        elif args.pretrained_model in ["codebert", "graphcodebert"]:
            decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=args.num_attention_heads)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
            self.model = Seq2Seq_bert(encoder=basemodel, decoder=decoder, config=config, args = args,
                                 beam_size=args.beam_size, max_length=args.max_output_tokens,
                                 sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    def forward(self, diff_ids, diff_mask, msg_ids, msg_mask):
        if self.args.pretrained_model in ["codet5", "plbart", "plbart-large"]:
            outputs = self.basemodel(input_ids=diff_ids, attention_mask=diff_mask,
                                     labels=msg_ids, decoder_attention_mask=msg_mask)
            loss = outputs.loss
        elif self.args.pretrained_model in ["unixcoder"]:
            outputs = self.model(source_ids=diff_ids, target_ids=msg_ids)
            loss = outputs[0]
        elif self.args.pretrained_model in ["codebert", "graphcodebert"]:
            outputs = self.model(source_ids=diff_ids, source_mask=diff_mask,
                                 target_ids=msg_ids, target_mask=msg_mask)
            loss = outputs[0]

        return loss

    def generate_preds(self, diff_ids, diff_mask):
        top_preds = []
        if self.args.pretrained_model in ["codet5", "plbart", "plbart-large"]:
            preds = self.basemodel.generate(input_ids=diff_ids, attention_mask=diff_mask)
            top_preds = list(preds.cpu().numpy())
        elif self.args.pretrained_model in ["unixcoder"]:
            preds = self.model(source_ids=diff_ids)
            top_preds = [pred[0].cpu().numpy() for pred in preds]
        elif self.args.pretrained_model in ["codebert", "graphcodebert"]:
            preds = self.model(source_ids=diff_ids, source_mask=diff_mask)
            top_preds = [pred[0].cpu().numpy() for pred in preds]

        return top_preds

