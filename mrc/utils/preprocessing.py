import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from configs.config import *
from utils.dataset import *
from models.model import *


RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


DATA_DIR= './Portfolio/mrc/data'

train_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'train.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'train')
val_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'train.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'val')


train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, 
                              shuffle=True,
                              pin_memory=PIN_MEMORY,
                              drop_last=DROP_LAST)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, 
                            shuffle=False,
                            pin_memory=PIN_MEMORY,
                            drop_last=DROP_LAST)

print(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}")


test_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'test.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'test')

question_ids = test_dataset.question_ids

test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, 
                            shuffle=False,
                            pin_memory=PIN_MEMORY,
                            drop_last=DROP_LAST)