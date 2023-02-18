import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.append('/Portfolio/speaker-emotion/configs')
from config import CFG


tokenizers = AutoTokenizer.from_pretrained(CFG["PLM"])

class CustomDataset(Dataset):
    
    def __init__(self, data, mode = "train"):
        self.dataset = data
        self.tokenizer = tokenizers
        self.mode = mode
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset['Utterance'][idx]
        inputs = self.tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
    
        if self.mode == "train":
            y = self.dataset['Target'][idx]
            return input_ids, attention_mask, y
        
        else:
            return input_ids, attention_mask