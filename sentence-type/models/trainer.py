import sys
import gc
import time
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back, Style

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAccuracy

from configs.config import config
from utils.preprocessing import *
from models.model import *
from train import *


b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL


loss_fn = {'type': nn.NLLLoss().to(config['device']),
           'pn' : nn.NLLLoss().to(config['device']),
           'time': nn.NLLLoss().to(config['device']),
           'sure': nn.BCELoss().to(config['device'])}


optimizer = torch.optim.AdamW(model.parameters(),
                              lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])


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


def train_one_epoch(model, 
                    dataloader, 
                    optimizer,
                    scheduler,
                    epoch,
                    device = config['device']):


    ################ torchmetrics: initialize metric #########################
    metric_acc_type = MulticlassAccuracy(average = 'weighted', num_classes = 4).to(config['device'])
    metric_f1_type = MulticlassF1Score(average = 'weighted', num_classes=4).to(config['device'])

    metric_acc_pn = MulticlassAccuracy(average = 'weighted', num_classes = 3).to(config['device'])
    metric_f1_pn = MulticlassF1Score(average = 'weighted', num_classes = 3).to(config['device'])

    metric_acc_time = MulticlassAccuracy(average = 'weighted', num_classes = 3).to(config['device'])
    metric_f1_time = MulticlassF1Score(average = 'weighted', num_classes=3).to(config['device'])

    metric_bi_acc = BinaryAccuracy().to(config['device'])
    metric_bi_f1 = BinaryF1Score().to(config['device'])
    
    ############################################################################

    train_loss = 0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total = len(dataloader))

    model.train()
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        masks = data['attention_mask'].to(device, dtype = torch.long)
        # targets
        t_type = data['target_type'].to(device, dtype = torch.long)
        t_pn = data['target_pn'].to(device, dtype = torch.long)
        t_time = data['target_time'].to(device, dtype = torch.long)
        t_sure = data['target_sure'].to(device, dtype = torch.float)
 
        # y_preds
        y_preds = model(ids, masks) 

        # Loss
        loss1 = loss_fn['type'](y_preds['type'], t_type )
        loss2 = loss_fn['pn'](y_preds['pn'], t_pn )
        loss3 = loss_fn['time'](y_preds['time'], t_time )
        y_preds['sure'] = y_preds['sure'].view(-1)
        loss4 = loss_fn['sure'](y_preds['sure'], t_sure )

        # loss sum
        loss = (loss1 + loss2 + loss3 + loss4) / 4

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        batch_size = ids.size(0)
        dataset_size += batch_size
        train_loss += float(loss.item() * batch_size) 
        train_epoch_loss = train_loss / dataset_size 
        
        # Type - ACC, F1
        acc_type = metric_acc_type(y_preds['type'], t_type )
        acc_type = acc_type.detach().cpu().item()
        f1_type = metric_f1_type(y_preds['type'], t_type )
        f1_type = f1_type.detach().cpu().item()

        # PN - ACC, F1
        acc_pn = metric_acc_pn(y_preds['pn'], t_pn )
        acc_pn = acc_pn.detach().cpu().item()
        f1_pn = metric_f1_pn(y_preds['pn'], t_pn )
        f1_pn = f1_pn.detach().cpu().item()

        # TIME - ACC, F1
        acc_time = metric_acc_time(y_preds['time'], t_time )
        acc_time = acc_time.detach().cpu().item()
        f1_time = metric_f1_time(y_preds['time'], t_time )
        f1_time = f1_time.detach().cpu().item()

        # SURE - ACC, F1
        acc_sure = metric_bi_acc(y_preds['sure'], t_sure )
        acc_sure = acc_sure.detach().cpu().item()
        f1_sure = metric_bi_f1(y_preds['sure'], t_sure )
        f1_sure = f1_sure.detach().cpu().item()

        bar.set_postfix(Epoch = epoch, 
                        Train_loss = train_epoch_loss,
                        LR = optimizer.param_groups[0]['lr'])
        
    # Type - ACC, F1
    acc_type = metric_acc_type.compute()
    f1_type = metric_f1_type.compute()

    # PN - ACC, F1
    acc_pn = metric_acc_pn.compute()
    f1_pn = metric_f1_pn.compute()

    # TIME - ACC, F1
    acc_time = metric_acc_time.compute()
    f1_time = metric_f1_time.compute()

    # SURE - ACC, F1
    acc_sure = metric_bi_acc.compute()
    f1_sure = metric_bi_f1.compute()


    print(f"Train | Type' | PosNg | Time' | Sure' |")
    print(f"ACCUR | {acc_type:.3f} | {acc_pn:.3f} | {acc_time:.3f} | {acc_sure:.3f} |")
    print(f"F1_SC | {f1_type:.3f} | {f1_pn:.3f} | {f1_time:.3f} | {f1_sure:.3f} |")


    acc_metric = {'type': acc_type, 'pn' : acc_pn,'time': acc_time,'sure': acc_sure}
    f1_metric = {'type': f1_type, 'pn' : f1_pn,'time': f1_time,'sure': f1_sure}

    del acc_type, acc_pn, acc_time, acc_sure, f1_type, f1_pn, f1_time, f1_sure
 
    metric_acc_type.reset()
    metric_acc_pn.reset()
    metric_acc_time.reset()
    metric_bi_acc.reset()

    metric_bi_acc.reset()
    metric_f1_pn.reset()
    metric_f1_time.reset()
    metric_bi_f1.reset()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return train_epoch_loss, acc_metric, f1_metric


@torch.no_grad()
def valid_one_epoch(model, 
                    dataloader, 
                    epoch, 
                    device = config['device']):
    

    ################ torchmetrics: initialize metric #########################
    metric_acc_type = MulticlassAccuracy(average = 'weighted', num_classes = 4).to(config['device'])
    metric_f1_type = MulticlassF1Score(average = 'weighted', num_classes=4).to(config['device'])

    metric_acc_pn = MulticlassAccuracy(average = 'weighted', num_classes = 3).to(config['device'])
    metric_f1_pn = MulticlassF1Score(average = 'weighted', num_classes = 3).to(config['device'])

    metric_acc_time = MulticlassAccuracy(average = 'weighted', num_classes = 3).to(config['device'])
    metric_f1_time = MulticlassF1Score(average = 'weighted', num_classes=3).to(config['device'])

    metric_bi_acc = BinaryAccuracy().to(config['device'])
    metric_bi_f1 = BinaryF1Score().to(config['device'])
    
    ############################################################################
    
    valid_loss = 0
    dataset_size = 0
    
    bar = tqdm(enumerate(dataloader), total = len(dataloader))

    model.eval()
    with torch.no_grad():
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

            # targets
            t_type = data['target_type'].to(device, dtype = torch.long)
            t_pn = data['target_pn'].to(device, dtype = torch.long)
            t_time = data['target_time'].to(device, dtype = torch.long)
            t_sure = data['target_sure'].to(device, dtype = torch.float)
    
            # y_preds
            y_preds = model(ids, masks) 

            # Loss
            loss1 = loss_fn['type'](y_preds['type'], t_type )
            loss2 = loss_fn['pn'](y_preds['pn'], t_pn )
            loss3 = loss_fn['time'](y_preds['time'], t_time )
            y_preds['sure'] = y_preds['sure'].view(-1)
            loss4 = loss_fn['sure'](y_preds['sure'], t_sure )

            # loss sum
            loss = (loss1 + loss2 + loss3 + loss4) / 4

            # 실시간 Loss
            batch_size = ids.size(0)
            dataset_size += batch_size
            valid_loss += float(loss.item() * batch_size)
            valid_epoch_loss = valid_loss / dataset_size

            # Type - ACC, F1
            acc_type = metric_acc_type(y_preds['type'], t_type )
            acc_type = acc_type.detach().cpu().item()
            f1_type = metric_f1_type(y_preds['type'], t_type )
            f1_type = f1_type.detach().cpu().item()

            # PN - ACC, F1
            acc_pn = metric_acc_pn(y_preds['pn'], t_pn )
            acc_pn = acc_pn.detach().cpu().item()
            f1_pn = metric_f1_pn(y_preds['pn'], t_pn )
            f1_pn = f1_pn.detach().cpu().item()

            # TIME - ACC, F1
            acc_time = metric_acc_time(y_preds['time'], t_time )
            acc_time = acc_time.detach().cpu().item()
            f1_time = metric_f1_time(y_preds['time'], t_time )
            f1_time = f1_time.detach().cpu().item()

            # SURE - ACC, F1
            acc_sure = metric_bi_acc(y_preds['sure'], t_sure )
            acc_sure = acc_sure.detach().cpu().item()
            f1_sure = metric_bi_f1(y_preds['sure'], t_sure )
            f1_sure = f1_sure.detach().cpu().item()

            bar.set_postfix(Epoch = epoch, 
                            Valid_loss = valid_epoch_loss,
                            LR = optimizer.param_groups[0]['lr'],
                            )

    # Type - ACC, F1
    acc_type = metric_acc_type.compute()
    f1_type = metric_f1_type.compute()

    # PN - ACC, F1
    acc_pn = metric_acc_pn.compute()
    f1_pn = metric_f1_pn.compute()

    # TIME - ACC, F1
    acc_time = metric_acc_time.compute()
    f1_time = metric_f1_time.compute()

    # SURE - ACC, F1
    acc_sure = metric_bi_acc.compute()
    f1_sure = metric_bi_f1.compute()


    print(f"Valid | Type' | PosNg | Time' | Sure' |")
    print(f"ACCUR | {acc_type:.3f} | {acc_pn:.3f} | {acc_time:.3f} | {acc_sure:.3f} |")
    print(f"F1_SC | {f1_type:.3f} | {f1_pn:.3f} | {f1_time:.3f} | {f1_sure:.3f} |")
    print()

    acc_metric = {'type': acc_type, 'pn' : acc_pn,'time': acc_time,'sure': acc_sure}
    f1_metric = {'type': f1_type, 'pn' : f1_pn,'time': f1_time,'sure': f1_sure}

    del acc_type, acc_pn, acc_time, acc_sure, f1_type, f1_pn, f1_time, f1_sure
    
    metric_acc_type.reset()
    metric_acc_pn.reset()
    metric_acc_time.reset()
    metric_bi_acc.reset()

    metric_bi_acc.reset()
    metric_f1_pn.reset()
    metric_f1_time.reset()
    metric_bi_f1.reset()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return valid_epoch_loss, acc_metric, f1_metric


# How to get Avg F1 Score
f1_metric_sample = {'type': 98.223, 'pn' : 82.334, 'time': 88.1, 'sure': 99.1111}
sum(f1_metric_sample.values()) / len(f1_metric_sample.values())


def run_training(model, optimizer, scheduler, device, n_epochs, fold):

    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()

    lowest_epoch = np.inf
    lowest_loss = np.inf

    
    best_score = 0
    best_score_epoch = np.inf
    best_model = "None"

    for epoch in range(1, n_epochs +1):
        gc.collect()

        train_epoch_loss, train_acc_metric, train_f1_metric = train_one_epoch(model= model, dataloader = train_loader, optimizer = optimizer,
                                                                              scheduler = scheduler, device = config['device'], epoch = epoch)
                                                                                        
        valid_epoch_loss, valid_acc_metric, valid_f1_metric = valid_one_epoch(model, dataloader = valid_loader,
                                                                              device = config['device'], epoch = epoch)
        # Mean Weighted F1_score
        train_f1 = sum(train_f1_metric.values()) / len(train_f1_metric.values())
        valid_f1 = sum(valid_f1_metric.values()) / len(valid_f1_metric.values())
        
        print()
        print(f"Epoch:{epoch:02d} | TL:{train_epoch_loss:.3e} | VL:{valid_epoch_loss:.3e} | Train's F1: {train_f1:.3f} | Valid's F1: {valid_f1:.3f} | ")
        print()

        if valid_epoch_loss < lowest_loss:
            print(f"{b_}Validation Loss Improved({lowest_loss:.3e}) --> ({valid_epoch_loss:.3e})")
            lowest_loss = valid_epoch_loss
            lowest_epoch = epoch
            PATH = model_save + f"Loss-Fold-{fold}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Better Loss Model Saved{sr_}")

        if best_score < valid_f1:
            print(f"{b_}F1 Improved({best_score:.3f}) --> ({valid_f1:.3f})")
            best_score = valid_f1
            best_score_epoch = epoch
            PATH2 = model_save + f"Loss-Fold-{fold}_f1.bin"
            torch.save(model.state_dict(), PATH2)
            print(f"Better_F1_Model Saved{sr_}")
            
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss : %.4e at %d th Epoch of %dth Fold" % (lowest_loss, lowest_epoch, fold))
    print("Best F1(W): %.4f at %d th Epoch of %dth Fold" % (best_score, best_score_epoch, fold))
    
    return model, best_score


test_ds = MyDataset(test, 
                    tokenizer = config['tokenizer'],
                    max_length = config['max_length'],
                    mode = "test")


test_loader = DataLoader(test_ds,
                         batch_size = config['valid_batch_size'],
                         num_workers = 2,
                         collate_fn = collate_fn,
                         shuffle = False, 
                         pin_memory = True, 
                         drop_last= False)

@torch.no_grad()
def test_func(model, dataloader = test_loader, device = config['device']):

    type_preds, pn_preds, time_preds, sure_preds = [], [], [], []

    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total = len(dataloader))
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

            # y_preds
            y_preds = model(ids, masks) 

            type_preds.append(y_preds['type'].detach().cpu().numpy())
            pn_preds.append(y_preds['pn'].detach().cpu().numpy())
            time_preds.append(y_preds['time'].detach().cpu().numpy())
            sure_preds.append(y_preds['sure'].detach().cpu().numpy())

    predictions = dict()

    type_predict = np.concatenate(type_preds, axis= 0)
    pn_predict = np.concatenate(pn_preds, axis= 0)
    time_predict = np.concatenate(time_preds, axis= 0)
    sure_predict = np.concatenate(sure_preds, axis= 0)

    predictions['type'] = type_predict
    predictions['pn'] = pn_predict
    predictions['time'] = time_predict
    predictions['sure'] = sure_predict

    gc.collect()
    
    return predictions