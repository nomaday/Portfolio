import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timezone, timedelta
import multiprocessing

from configs.config import CFG
from models.model import *
from models.trainer import *
from utils.dataset import CustomDataset, tokenizers
from utils.preprocess import *
from utils.metrics import competition_metric


# 시간 고유값 
PROJECT_DIR = './'
os.chdir(PROJECT_DIR)
kst = timezone(timedelta(hours=9))        
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', train_serial)

# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)    

def run():
    multiprocessing.freeze_support()

    for fold in range(0, CFG['NFOLD']):
        print(f"======== Fold: {fold} =========")

        model = BaseModel().to(device)
        # model.eval()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])

        infer_model = trainer(model, optimizer, train_dataloader, val_dataloader, device, fold)

if __name__ == '__main__':
    run()