import os
from sklearn.metrics import accuracy_score
from datetime import datetime, timezone, timedelta
import multiprocessing

from transformers import ElectraTokenizerFast
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from configs.config import *
from models.model import *
from utils.util import *
from utils.preprocessing import *
from models.trainer import *


model = electra(pretrained="monologg/koelectra-base-v3-discriminator").to(device)

optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
loss = F.cross_entropy
metrics = accuracy_score

tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  device=device,
                  tokenizer=tokenizer,
                  interval=LOGGING_INTERVAL)


early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE,
                            mode=min)


kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

PROJECT_DIR = './Portfolio/mrc'

RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)


recorder = Recorder(record_dir=RECORDER_DIR,
                    model=model,
                    optimizer=optimizer)


# Train

def run():
    multiprocessing.freeze_support()

    for epoch_index in range(EPOCHS):

        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial

        """
        Train
        """
        print(f"Train {epoch_index}/{EPOCHS}")
        print(f"--Train {epoch_index}/{EPOCHS}")
        trainer.train(dataloader=train_dataloader, epoch_index=epoch_index, tokenizer=tokenizer, mode='train')

        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time 

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score
        trainer.clear_history()

        """
        Validation
        """
        print(f"Val {epoch_index}/{EPOCHS}")
        print(f"--Val {epoch_index}/{EPOCHS}")
        trainer.train(dataloader=val_dataloader, epoch_index=epoch_index, tokenizer=tokenizer, mode='val')

        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time 

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_{metric_str}"] = score
        trainer.clear_history()

        """
        Record
        """
        recorder.add_row(row_dict)

        """
        Early stopper
        """
        early_stopping_target = EARLY_STOPPING_TARGET
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)

        if early_stopper.stop == True:
            print(f"Early stopped, counter {early_stopper.patience_counter}/{EARLY_STOPPING_PATIENCE}")

            break

if __name__ == '__main__':
    run()