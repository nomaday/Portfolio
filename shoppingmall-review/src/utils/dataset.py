import torch 
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.reviews = df['reviews']
        self.target = df['target']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        reviews = self.reviews[index]
        inputs = self.tokenizer.encode_plus(
            reviews,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
            )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(self.target[index], dtype=torch.long)
            }
    

class DatasetInfer(Dataset):
    
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.reviews = df['reviews']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        reviews = self.reviews[index]
        inputs = self.tokenizer.encode_plus(
            reviews,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
            )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }