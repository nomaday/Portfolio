#!pip install pytest-warnings
#!pip install colorama

import os
import gc
import torch
import numpy as np
import pandas as pd
import warnings
import sys
from configs.config import *

#from google.colab import drive
#drive.mount('/content/drive')

from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

import multiprocessing

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from src.utils.preprocessing import prepare_loader
from src.models.model import *
from src.models.trainer import run_training, CosineWarmupScheduler


# Suppress warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Run Train
def run():
    multiprocessing.freeze_support()
    for fold in range(0, config['n_folds']):

        print(f"{y_}==== Fold: {fold} ====={sr_}")
        train_loader, valid_loader = prepare_loader(fold = fold)

        model = Model(config['model'])
        model = model.to(config['device'])

        optimizer = AdamW(model.parameters(),
                        lr = config['learning_rate'],
                        weight_decay = config['weight_decay'])
        
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)

        model, train_histories, valid_histories = run_training(model, 
                                                        optimizer,
                                                        scheduler,
                                                        device = config['device'],
                                                        n_epochs = config['n_epochs'],
                                                        fold = fold
                                                        )   
        
        ## 메모리 절약
        del model, train_histories, train_loader, valid_loader, valid_histories

        _ = gc.collect()
        
        print()

if __name__ == '__main__':
    run()