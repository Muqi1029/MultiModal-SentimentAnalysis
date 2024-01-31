import torch
from torch import nn
from transformers import AutoModel


class TextModel(nn.Module):
    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config['bert_name'])
        self.trans = nn.Sequential(
            nn.Dropout(config['bert_dropout']),
            nn.Linear(self.bert.config.hidden_size, config['hidden_dim']),
            nn.ReLU(inplace=True)
        )

        for param in self.bert.parameters():
            if config['finetune']:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, bert_inputs, masks, token_type_ids=None, attention=False):
        assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        pooler_out = bert_out['pooler_output']
        if attention:
            return self.trans(bert_out['last_hidden_state']), self.trans(pooler_out)
        return self.trans(pooler_out)
