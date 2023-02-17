#!pip install transformers

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig
import sys
from configs.config import *
from src.utils.preprocessing import *

# Tokenizer
config['tokenizer'] = AutoTokenizer.from_pretrained(config['model'])

# 확인
print(config['tokenizer'].tokenize(train['reviews'][0]))
print(config['tokenizer'](train['reviews'][0]))


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Model(nn.Module):

    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(config['model'])
        self.config = AutoConfig.from_pretrained(config['model'])
        self.drop = nn.Dropout(p=0.25)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(768, config['output_dim'])
        self.logsoftmax = nn.LogSoftmax(dim = -1)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,
                         attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        outputs = self.fc(out)
        outputs = self.logsoftmax(outputs)

        return outputs

model = Model(config['model'])
model = model.to(config['device'])