import os
import sys
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel

from configs.config import CFG
from utils.dataset import CustomDataset, tokenizers


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizers = AutoTokenizer.from_pretrained(CFG["PLM"])


class BaseModel(nn.Module):

    def __init__(self, dropout=0.5, num_classes= 7 ):

        super().__init__()

        self.bert = AutoModel.from_pretrained(CFG["PLM"])

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,
                                     return_dict=False
                                    )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer