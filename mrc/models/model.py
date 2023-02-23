import os
import numpy as np
from transformers import ElectraForQuestionAnswering
import torch
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class electra(nn.Module):

    def __init__(self, pretrained, **kwargs):
        super(electra, self).__init__()

        self.model = ElectraForQuestionAnswering.from_pretrained(pretrained)


    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        
        outputs = self.model(input_ids = input_ids, 
                             attention_mask = attention_mask,
                             start_positions = start_positions,
                             end_positions = end_positions)
        
        return outputs


class EarlyStopper():

    def __init__(self, patience: int, mode:str)-> None:
        self.patience = patience
        self.mode = mode

        # Initiate
        self.patience_counter = 0
        self.stop = False
        self.best_loss = np.inf

        print(f"Initiated early stopper, mode: {self.mode}, best score: {self.best_loss}, patience: {self.patience}")

        
    def check_early_stopping(self, loss: float)-> None:
        loss = -loss if self.mode == 'max' else loss  # get max value if mode set to max

        if loss > self.best_loss:
            # got worse score
            self.patience_counter += 1

            print(f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{abs(self.best_loss)} -> now:{abs(loss)}")
            
            if self.patience_counter == self.patience:
                print(f"Early stopper, stop")
                self.stop = True  # end

        elif loss <= self.best_loss:
            # got better score
            self.patience_counter = 0
            
            print(f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{abs(self.best_loss)} -> now:{abs(loss)}")
            print(f"Set counter as {self.patience_counter}")
            print(f"Update best score as {abs(loss)}")
            
            self.best_loss = loss
            
        else:
            print('debug')