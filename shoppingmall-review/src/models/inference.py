import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc
import time
import copy
from copy import deepcopy
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL
from sklearn.preprocessing import LabelEncoder
import sys
from configs.config import *
from src.models.model import *
from src.utils.preprocessing import *
from src.utils.preprocessing import test_loader, ss, encoder
from src.utils.metrics import calc_accuracy
from src.models.trainer import test_func


base_path = '/Portfolio/shoppingmall-review/result/'

model_paths = [
    base_path + "model/Loss-Fold-0.bin",
    base_path + "model/Loss-Fold-1.bin",
    base_path + "model/Loss-Fold-2.bin",
    base_path + "model/Loss-Fold-3.bin",
    base_path + "model/Loss-Fold-4.bin",
    base_path + "model/Loss-Fold-5.bin",
    base_path + "model/Loss-Fold-6.bin", 
    base_path + "model/Loss-Fold-7.bin"
    ]


def inference(model_paths, dataloader, device=config['device']):

    final_preds = []

    for i, path in enumerate(model_paths):
        model = Model(config['model'])
        model.to(config['device'])
        model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        preds = test_func(model, dataloader, device)
        final_preds.append(preds)
    
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


preds = inference(model_paths, test_loader, config['device'])

print(preds.shape) 

# argmax로 target class 확인
new_preds = np.argmax(preds, axis = 1) 

# 먼저 Target Class를 Submission CSV 파일에 넣습니다.
ss['target'] = new_preds

# Value Counts()로 Class 별 수치 확인
ss.target.value_counts()

# Encoded Target을 원래대로 돌립니다. 
ss['target'] = encoder.inverse_transform(ss.target)
print(ss.shape)

ss.target.value_counts() 


# Submission CSV file Save
submission_path = base_path + "submission_csv/"
ss.to_csv(submission_path + "result_submission.csv", index=False) 