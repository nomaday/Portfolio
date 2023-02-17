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
import sys
from configs.config import *
from src.models.model import *
from src.utils.preprocessing import prepare_loader
from src.utils.metrics import calc_accuracy


# Loss Function
loss_fn = nn.NLLLoss().to(config['device'])


# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=config['learning_rate'], #5e-5
                              weight_decay=config['weight_decay']
                              )

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

# Train One Epoch Function
def train_one_epoch(model, 
                    dataloader, 
                    optimizer,
                    scheduler,
                    epoch,
                    device = config['device']):
    
    y_true = []
    preds = []

    train_loss = 0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total = len(dataloader))

    model.train()

    for step, data in bar:
        ids = data['ids'].to(device, dtype = torch.long)
        masks = data['mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)

        batch_size = ids.size(0)

        y_preds = model(ids, masks)
        loss = loss_fn(y_preds, targets)

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        dataset_size += batch_size
        train_loss += float(loss.item() * batch_size) 

        train_loss /= dataset_size 

        preds.append(y_preds)
        y_true.append(targets)

        bar.set_postfix(Epoch = epoch, 
                        Train_loss = train_loss,
                        LR = optimizer.param_groups[0]['lr']
                        )
        
    preds = torch.cat(preds, dim = 0)
    y_true = torch.cat(y_true, dim = 0)

    # Accuracy
    accuracy = calc_accuracy(preds, y_true)
    print()
    print("Train's Accuracy: %.2f percent" % accuracy)
    print()
    gc.collect()

    return train_loss


# Valid One Epoch Function
@torch.no_grad()

def valid_one_epoch(model, 
                    dataloader, 
                    epoch, 
                    device = config['device']):
    
    y_true = []
    preds = []
    
    valid_loss = 0
    dataset_size = 0
    
    bar = tqdm(enumerate(dataloader), total = len(dataloader))

    # 위 Annotation이 있지만,
    model.eval()

    with torch.no_grad():
        for step, data in bar:
            ids = data['ids'].to(device, dtype = torch.long)
            masks = data['mask'].to(device, dtype = torch.long)
            targets = data['target'].to(device, dtype = torch.long)

            batch_size = ids.size(0)

            y_preds = model(ids, masks)
            loss = loss_fn(y_preds, targets)

            dataset_size += batch_size
            valid_loss += float(loss.item() * batch_size)
            valid_loss /= dataset_size

            preds.append(y_preds)
            y_true.append(targets)

            bar.set_postfix(Epoch = epoch, 
                            Valid_loss = valid_loss,
                            LR = optimizer.param_groups[0]['lr']
                            )
    
    preds = torch.cat(preds, dim = 0)
    y_true = torch.cat(y_true, dim = 0)

    accuracy = calc_accuracy(preds, y_true)
    print()
    print("Valid's Accuracy: : %.2f precent" % accuracy)
    print()
    gc.collect()

    return valid_loss


# Test Function
@torch.no_grad()
def test_func(model, dataloader, device = config['device']):
    preds= []

    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total = len(dataloader))
        for step, data in bar:
            ids = data['ids'].to(device, dtype = torch.long)
            masks = data['mask'].to(device, dtype = torch.long)

            y_preds = model(ids, masks)
            preds.append(y_preds.detach().cpu().numpy())

    predictions = np.concatenate(preds, axis= 0)
    gc.collect()

    return predictions


# Run Training Function
base_path = '/Portfolio/shoppingmall-review/result/'


def run_training(model, optimizer, scheduler, device, n_epochs, fold):

    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    lowest_epoch = np.inf
    lowest_loss = np.inf
    train_history, valid_history = [],  []

    for epoch in range(1, n_epochs +1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model= model,
                                           dataloader = train_loader,
                                           optimizer = optimizer,
                                           scheduler = scheduler,
                                           device = config['device'],
                                           epoch = epoch
                                           )
        
        valid_epoch_loss = valid_one_epoch(model,
                                           dataloader = valid_loader,
                                           device = config['device'],
                                           epoch = epoch)
        
        train_history += [train_epoch_loss]
        valid_history += [valid_epoch_loss]

        if valid_epoch_loss <= lowest_loss:
            print(f"{b_}Validation Loss Improved({lowest_loss}) --> ({valid_epoch_loss})")
            lowest_loss = valid_epoch_loss
            lowest_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = base_path + f"model/Loss-Fold-{fold}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{sr_}")
        
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: %.4e at %d th Epoch of %dth Fold" % (lowest_loss, lowest_epoch, fold))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_history, valid_history