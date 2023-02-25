import numpy as np
from tqdm import tqdm
import os
import sys

from model import *
from configs.config import CFG
from utils.dataset import CustomDataset, tokenizers
from utils.metrics import competition_metric
from train import *

def trainer(model, optimizer, train_loader, test_loader, device, fold=CFG["NFOLD"]):

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    best_score = 0
    best_model = "None"
    for epoch_num in range(CFG["EPOCHS"]):

        model.train()
        train_loss = []
        for input_ids, attention_mask, train_label in tqdm(train_loader):

            optimizer.zero_grad()

            train_label = train_label.to(device)
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            output = model(input_id, mask)     
    
            batch_loss = criterion(output, train_label.long()) 
            train_loss.append(batch_loss.item())
            
            batch_loss.backward()
            optimizer.step()

        val_loss, val_score = validation(model, criterion, test_loader, device)
        print(f'Epoch [{epoch_num}] of {fold}th Fold, Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')

        if best_score < val_score:
            best_model = model
            best_score = val_score
            torch.save(model.state_dict(), os.path.join(RECORDER_DIR, f"best_model-Fold-{fold}.pt"))
        
    return best_model         


def validation(model, criterion, test_loader, device):
    model.eval()

    val_loss = []
    model_preds = []
    true_labels = []  
    with torch.no_grad():
        for input_ids, attention_mask, valid_label in tqdm(test_loader):
            
            valid_label = valid_label.to(device)
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            output = model(input_id, mask)     
    
            batch_loss = criterion(output, valid_label.long()) 
            val_loss.append(batch_loss.item())      
            
            model_preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += valid_label.detach().cpu().numpy().tolist()
        val_f1 = competition_metric(true_labels, model_preds)
    return val_loss, val_f1