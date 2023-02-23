import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

from configs.config import config


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


config['tokenizer'] = AutoTokenizer.from_pretrained(config['model'])


class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_name,  attention_type="original_full")
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()

        # TYPE: 유형
        self.fc_type = nn.Linear(self.config.hidden_size, 4)
        self.logsoftmax_type = nn.LogSoftmax(dim = -1)

        # PN: 극성
        self.fc_pn = nn.Linear(self.config.hidden_size, 3)
        self.logsoftmax_pn = nn.LogSoftmax(dim = -1)

        # TIME: 시제
        self.fc_time = nn.Linear(self.config.hidden_size, 3)
        self.logsoftmax_time = nn.LogSoftmax(dim = -1)

        # SURE: 확실성 - Binary Classification
        self.fc_sure = nn.Linear(self.config.hidden_size, 1)
        self.sigmoid_sure = nn.Sigmoid()
        

    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)

        # TYPE: 유형
        out_type = self.fc_type(out)
        out_type = self.logsoftmax_type(out_type)

        # PN: 극성
        out_pn = self.fc_pn(out)
        out_pn = self.logsoftmax_pn(out_pn)

        # TIME: 시제
        out_time = self.fc_time(out)
        out_time = self.logsoftmax_time(out_time)

        # SURE: 확실성 - Binary Classification
        out_sure = self.fc_sure(out)
        out_sure = self.sigmoid_sure(out_sure)
        
        outputs = {'type': out_type, 'pn': out_pn, 'time': out_time, 'sure': out_sure}

        return outputs


model = Model(config['model'])
model = model.to(config['device'])