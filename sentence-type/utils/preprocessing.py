import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DataCollatorWithPadding

import torch 
from torch.utils.data import Dataset, DataLoader

from configs.config import config


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(config['seed'])

data_path = './Portfolio/sentence-type/data/'

train = pd.read_csv(data_path + "train.csv")
test = pd.read_csv(data_path + "test.csv")
ss = pd.read_csv(data_path + "sample_submission.csv")

print("Train Shape: ", train.shape, "Test Shape: ", test.shape)

# Column Rename
train.rename(columns={'문장':'new_sentence'}, inplace = True)
test.rename(columns={'문장':'new_sentence'}, inplace = True)

# 라벨 인코딩
for target in train.columns[2:-2]:
    print(f"{target}: ", f"{train[target].unique()}")

target1_encode = {v: k for k, v in enumerate(train["유형"].unique())}
target1_inverse = {v: k for k, v in target1_encode.items()}
train['type'] = train["유형"].apply(lambda x: target1_encode[x])
train["유형"].value_counts()
train["type"].value_counts()

target2_encode = {v: k for k, v in enumerate(train["극성"].unique())}
target2_inverse = {v: k for k, v in target2_encode.items()}
train['pn'] = train["극성"].apply(lambda x: target2_encode[x])
train["극성"].value_counts()
train["pn"].value_counts()

target3_encode = {v: k for k, v in enumerate(train["시제"].unique())}
target3_inverse = {v: k for k, v in target3_encode.items()}
train['time'] = train["시제"].apply(lambda x: target3_encode[x])
train["시제"].value_counts()
train["time"].value_counts()

train['sure'] = train["확실성"].apply(lambda x: 1 if x == '확실' else 0)
train["확실성"].value_counts()
train["sure"].value_counts()


train["label"].value_counts()

train.drop(['ID', '유형', '극성', '시제', '확실성', 'label'], axis = 1, inplace = True)

print(train.shape)


# Max Length
max_len = 0
lens = []
for index in range(train.shape[0]):
    lens2 = len(config['tokenizer'].tokenize(train.loc[index, "new_sentence"]))
    lens.append(lens2)
    if lens2 > max_len:
        max_len = lens2

np.mean(lens)

# 시각화
plt.figure(figsize = (12, 8))
sns.histplot(data= lens)

a = dict()
k = 0
for index in range(len(lens)):
    if lens[index] >= 128:
        if lens[index] not in a.keys():
            a[lens[index]] = 1
        else:
            a[lens[index]] += 1 

config['max_length']


# Collate_fn
collate_fn = DataCollatorWithPadding(tokenizer=config['tokenizer'] )


# MultilabelStratifiedKFold
config['target_cols'] = ['type', 'pn', 'time', 'sure']

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

skf = MultilabelStratifiedKFold(n_splits = config['n_folds'], shuffle = True, random_state = config['seed'])

for fold, (_, val_index) in enumerate(skf.split(X=train, y=train[config['target_cols']])):
    train.loc[val_index, 'kfold'] = int(fold)

train['kfold'] = train['kfold'].astype('int')
print(train.shape)


class MyDataset(Dataset):

    def __init__(self, 
                 df = train,
                 tokenizer = config['tokenizer'], 
                 max_length = config['max_length'], 
                 mode = "train"):
        self.df = df
        self.max_length=  max_length
        self.tokenizer = tokenizer
        self.mode = mode

        self.text = self.df.new_sentence.values
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.text[idx]
        inputs = self.tokenizer.encode_plus(sentence, 
                                            add_special_tokens=True,
                                            # padding='max_length', 
                                            max_length = self.max_length, 
                                            truncation=True)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if self.mode == "train":
            
            # y1: TYPE - 유형
            y_type = self.df['type'].values[idx]

            # y2:  PN - 극성
            y_pn = self.df['pn'].values[idx]

            # y3: time - 시제
            y_time = self.df['time'].values[idx]

            # y4: sure - 확실성
            y_sure = self.df['sure'].values[idx]


            return {'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'target_type': y_type,
                    'target_pn': y_pn,
                    'target_time': y_time,
                    'target_sure': y_sure}

        else:
            return {'input_ids': inputs['input_ids'], 
                    'attention_mask': inputs['attention_mask'],
                    }
        

def prepare_loader(fold):
    
    train_df = train[train.kfold != fold].reset_index(drop=True)
    valid_df = train[train.kfold == fold].reset_index(drop=True)

    ## train, valid -> Dataset
    train_ds = MyDataset(train_df, 
                       tokenizer = config['tokenizer'],
                       max_length = config['max_length'],
                       mode = "train")

    valid_ds = MyDataset(valid_df, 
                       tokenizer = config['tokenizer'],
                       max_length = config['max_length'],
                       mode = "train")
    
    # Dataset -> DataLoader
    train_loader = DataLoader(train_ds,
                              batch_size = config['train_batch_size'], 
                              collate_fn=collate_fn, 
                              num_workers = 2,
                              shuffle = True, 
                              pin_memory = True, 
                              drop_last= True)

    valid_loader = DataLoader(valid_ds,
                              batch_size = config['valid_batch_size'],
                              collate_fn=collate_fn,
                              num_workers = 2,
                              shuffle = False, 
                              pin_memory = True,)
    

    return train_loader, valid_loader

train_loader, valid_loader = prepare_loader(fold)