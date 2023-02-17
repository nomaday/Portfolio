import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import torch 
from torch.utils.data import DataLoader
import os
from configs.config import *
from src.utils.dataset import Dataset, DatasetInfer


# Set Seed
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


# Original data load
data_path = '/Portfolio/shoppingmall-review/data/'

train = pd.read_csv(data_path + "train.csv")
test= pd.read_csv(data_path + "test.csv")
ss = pd.read_csv(data_path + "sample_submission.csv")

print("Train Shape: ", train.shape, "Test Shape: ", test.shape)

# Target Class
train.target.unique()

# Target Categorical Encoding
encoder = LabelEncoder()
train['new_target'] = encoder.fit_transform(train['target'])

# Column 정리 및 Reset
print(train.shape)
print(test.shape)

train.drop(['id', 'target'], axis = 1, inplace = True) # Drop 
train.columns = ['reviews', 'target'] # rename columns

test.drop(['id',], axis = 1, inplace = True) # Drop 
test.columns = ['reviews'] # rename columns


# StratifiedKFold
skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])

for fold, ( _, val_) in enumerate(skf.split(X=train, y=train.target)):
    train.loc[val_ , "kfold"] = int(fold)

train["kfold"] = train["kfold"].astype(int)


# DataLoader
def prepare_loader(fold):

    ## 여기서 데이터를 n_folds 수 만큼 Split 해줍니다. 
    ## train 데이터와 Validation 데이터로!
    train_df = train[train.kfold != fold].reset_index(drop=True)
    valid_df = train[train.kfold == fold].reset_index(drop=True)

    ## train, valid -> Dataset
    train_ds = Dataset(train_df, 
                       tokenizer = config['tokenizer'],
                       max_length = config['max_length'])

    valid_ds = Dataset(valid_df, 
                       tokenizer = config['tokenizer'],
                       max_length = config['max_length'])
    
    ## Dataset -> DataLoader
    train_loader = DataLoader(train_ds,
                              batch_size = config['train_batch_size'],
                              num_workers = 2,
                              shuffle = True, 
                              pin_memory = True, 
                              drop_last= False)

    valid_loader = DataLoader(valid_ds,
                              batch_size = config['valid_batch_size'],
                              num_workers = 2,
                              shuffle = False, 
                              pin_memory = True, 
                              drop_last= False)
    
    return train_loader, valid_loader

train_loader, valid_loader = prepare_loader(fold)


# DataLoader for Inference
test_ds = DatasetInfer(test, 
                       tokenizer = config['tokenizer'], 
                       max_length = config['max_length'])

test_loader = DataLoader(test_ds,
                         batch_size = config['valid_batch_size'],
                         num_workers = 2,
                         shuffle = False, 
                         pin_memory = True, 
                         drop_last= False)