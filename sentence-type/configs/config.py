import torch
from transformers import AutoTokenizer

config ={
    'testnum' : 't0-1',
    #'dsinfo' : 'org_train',
    'model': 'monologg/kobigbird-bert-base', #"microsoft/deberta-v3-base",
    'learning_rate': 5e-5, 
    'seed': 2022,
    'n_folds' : 10,
    'n_epochs': 5,
    "train_batch_size": 32, #16, 
    "valid_batch_size": 32, #16, 
    "max_length": 128,
    "scheduler": 'CosineAnnealingLR', #'CosineAnnealingWarmRestarts', #'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 1e-6,
    "n_accumulate": 1,
    'target_cols': ['type', 'pn', 'time', 'sure'],
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

config['tokenizer'] = AutoTokenizer.from_pretrained(config['model'])