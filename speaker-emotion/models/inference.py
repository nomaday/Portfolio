import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from tqdm import tqdm

from model import *
from utils.dataset import CustomDataset, tokenizers
from train import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base_path = f'./Portfolio/speaker-emotion/results/{train_serial}/'
print(base_path)

model_paths = [
    base_path + "best_model-Fold-0.pt",
    base_path + "best_model-Fold-1.pt",
    base_path + "best_model-Fold-2.pt",
    base_path + "best_model-Fold-3.pt",
    base_path + "best_model-Fold-4.pt",
    base_path + "best_model-Fold-5.pt",
    base_path + "best_model-Fold-6.pt",
    ]

test = pd.read_csv(path + 'test.csv')
test = CustomDataset(test, mode = "test")
test_dataloader = torch.utils.data.DataLoader(test, batch_size= 10, #CFG['BATCH_SIZE']
                                              shuffle=False)

def inference(model_paths, test_loader, device):

    test_predicts = []

    with torch.no_grad():

        for i, path in enumerate(model_paths):  
            test_predict = []
            
            model = BaseModel().to(device)
            model.load_state_dict(torch.load(path))
            model.eval()

            print(f"Prediction for model {i+1}")
            for input_ids, attention_mask in tqdm(test_loader):
                input_id = input_ids.to(device)
                mask = attention_mask.to(device)
                y_pred = model(input_id, mask)
                test_predict.append(y_pred.detach().cpu().numpy())

            test_predict1 = np.concatenate(np.array(test_predict), axis = 0) # test_predict1: [total_bs, 7]
            print(test_predict1.shape)
            
            test_predicts.append(test_predict1) # test_predicts: [[total_bs, 7],  [total_bs, 7],  .... ]

    test_predicts_final = np.mean(test_predicts, axis=0)
    
    return test_predicts_final


preds = inference(model_paths, test_dataloader, device)
n_preds = np.argmax(preds, axis = 1)
N_preds = le.inverse_transform(n_preds) 

submit = pd.read_csv(path + 'sample_submission.csv')
submit['Target'] = N_preds

submit.to_csv(f"./Portfolio/speaker-emotion/results/submit.csv", index=False)