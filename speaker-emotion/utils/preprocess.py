import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch

sys.path.append('/Portfolio/speaker-emotion/configs')
from config import CFG
sys.path.append('/Portfolio/speaker-emotion/utils')
from dataset import CustomDataset, tokenizers


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])


path = "/Portfolio/speaker-emotion/data/"
data = pd.read_csv(path + 'train.csv')

# Speaker 제외
train_ds = data[["ID", "Utterance", "Dialogue_ID", "Target"]]
train_ds["Target"].value_counts()

# Label encoding
le = LabelEncoder()
le = le.fit(train_ds['Target'])
train_ds['Target']=le.transform(train_ds['Target'])

for i, label in enumerate(le.classes_):
    print(i, '->', label)    

# Row Shuffling
train_ds = train_ds.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffling하고 index reset

# StratifiedKFold
skf = StratifiedKFold(n_splits=CFG['NFOLD'], shuffle=True, random_state=CFG['SEED'])

for fold, ( _, val_) in enumerate(skf.split(X=train_ds, y=train_ds.Target)):
    train_ds.loc[val_ , "Kfold"] = int(fold)

train_ds["Kfold"] = train_ds["Kfold"].astype(int)

CFG['NFOLD'], CFG['EPOCHS']

# ## Train/Validation split
train_df = train_ds[train_ds.Kfold != fold].reset_index(drop=True)
valid_df = train_ds[train_ds.Kfold == fold].reset_index(drop=True)

train_len=len(train_df)
val_len=len(valid_df)

print(train_len)
print(val_len)

train = CustomDataset(train_df, mode = "train")
valid = CustomDataset(valid_df, mode = "train")

train_dataloader = torch.utils.data.DataLoader(train, batch_size= CFG['BATCH_SIZE'], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valid, batch_size= CFG['BATCH_SIZE'], shuffle=False)