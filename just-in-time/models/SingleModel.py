import torch
import torch.nn as nn


class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.ll_proj = nn.Linear(args.hidden_size, 1)     # ll means last layer representations.

    def forward(self, features):
        cls_features = features[:, 0, :]

        cls_features = self.dropout(cls_features)
        proj_score = self.ll_proj(cls_features)
        return proj_score


class SingleModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(SingleModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = RobertaClassifier(args)

    def forward(self, input_ids, input_mask, manual_features, label):
        if self.args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask)
        elif self.args.pretrained_model in ["codet5"]:
            outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=input_mask)
        elif self.args.pretrained_model in ["plbart", "plbart-large"]:
            if self.args.use_lora:
                outputs = self.encoder.base_model.model.model.encoder(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = self.encoder.model.encoder(input_ids=input_ids, attention_mask=input_mask)

        logits = self.classifier(outputs[0])

        prob = torch.sigmoid(logits)

        loss_fct = nn.BCELoss()
        loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())

        return prob, loss
