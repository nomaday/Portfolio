import torch
from transformers import AutoTokenizer


config ={
    'model': "beomi/KcELECTRA-base", 
    'learning_rate':5e-5,
    'seed': 2022,
    'output_dim': 4,
    'n_folds' : 8,
    'n_epochs': 3,
    "train_batch_size": 64, #2*64,
    "valid_batch_size": 128, #2*64,
    "max_length": 128,
    "scheduler": 'CosineAnnealingLR', 
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 1e-6,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

config['tokenizer'] = AutoTokenizer.from_pretrained(config['model'])
