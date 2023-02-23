import os
import csv
import torch

from models.model import *


class Recorder():

    def __init__(self,
                 record_dir: str,
                 model: object,
                 optimizer: object):
        
        self.record_dir = record_dir
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')
        self.weight_path = os.path.join(record_dir, 'model.pt')

        self.model = model
        self.optimizer = optimizer

        
    def set_model(self, model: 'model'):
        self.model = model


    def add_row(self, row_dict: dict):

        fieldnames = list(row_dict.keys())

        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(row_dict)
            print(f"Write row {row_dict['epoch_index']}")

            
    def save_weight(self, epoch: int)-> None:
        check_point = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        torch.save(check_point, self.weight_path)
        print(f"Recorder, epoch {epoch} Model saved: {self.weight_path}")