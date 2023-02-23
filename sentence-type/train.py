import os
import gc
from datetime import datetime, timezone, timedelta
from colorama import Fore, Back, Style
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import AdamW

from configs.config import config
from utils.preprocessing import *
from models.model import *
from models.trainer import *


b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# 시간 고유값 
# PROJECT_DIR = './'
PROJECT_DIR = './Portfolio/sentence-type/'

os.chdir(PROJECT_DIR)
kst = timezone(timedelta(hours=9))        
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', train_serial + '_' + config['testnum'])

# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)  
print(f'RECORDER_DIR: {RECORDER_DIR}')

# 모델 저장 경로
model_save =  RECORDER_DIR +'/'
sub_path = RECORDER_DIR +'/'

print(f'model_save: {model_save}')
print(f'save_path: {sub_path}')


gc.collect()
torch.cuda.empty_cache()

# Run Train
best_scores = []

def run():
    multiprocessing.freeze_support()

    for fold in range(0, config['n_folds']):

        print(f"{y_}==== Fold: {fold} ====={sr_}")

        # DataLoaders
        train_loader, valid_loader = prepare_loader(fold = fold)

        # Define Model because of KFold
        model = Model(config['model'])
        model = model.to(config['device'])

        # Loss Function
        loss_fn = {'type': nn.NLLLoss().to(config['device']),
                    'pn' : nn.NLLLoss().to(config['device']),
                    'time': nn.NLLLoss().to(config['device']),
                    'sure': nn.BCELoss().to(config['device'])}

        # Define Opimizer and Scheduler
        optimizer = AdamW(model.parameters(),
                        lr = config['learning_rate'],
                        weight_decay = config['weight_decay'])
        
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)
        # scheduler = fetch_scheduler(optimizer)

        ## Start Training
        model, best_score = run_training(model, optimizer, scheduler, device = config['device'], n_epochs = config['n_epochs'], fold = fold)
        
        ## Best F1_Score per Fold 줍줍
        best_scores.append(best_score)
        
        ## For Memory
        del model, train_loader, valid_loader

        torch.cuda.empty_cache()
        _ = gc.collect()
        
        print()

if __name__ == '__main__':
    run()

print("Hey!")