import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from train import tokenizer


class QADataset(Dataset):
    
    def __init__ (self, data_dir: str, tokenizer, max_seq_len: int, mode = 'train'):
        self.mode = mode
        self.data = json.load(open(data_dir, 'r', encoding='utf8'))
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if mode == 'test':
            self.encodings, self.question_ids = self.preprocess()
        else:
            self.encodings, self.answers = self.preprocess()
        
    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, index: int):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}

    
    def preprocess(self):
        contexts, questions, answers, question_ids = self.read_squad()
        if self.mode == 'test':
            encodings = self.tokenizer(contexts, questions, truncation=True, max_length = self.max_seq_len, padding=True)
            return encodings, question_ids
        else: # train or val
            self.add_end_idx(answers, contexts)     # train.json에는 질문에 대한 답이 context 내에서 시작되는 index인 'answer_srart'만 있기 때문에, 추가로 'answer_end'를 찾아주는 함수
            encodings = self.tokenizer(contexts, questions, truncation=True, max_length = self.max_seq_len, padding=True)
            self.add_token_positions(encodings, answers)
        
            return encodings, answers
        
    
    def read_squad(self):
        contexts = []
        questions = []
        question_ids = []
        answers = []
        
        # train - val split
        if self.mode == 'train':
            self.data['data'] = self.data['data'][:-1*int(len(self.data['data'])*0.2)]
        elif self.mode == 'val':
            self.data['data'] = self.data['data'][-1*int(len(self.data['data'])*0.2):]
        
        
        till = len(self.data['data'])
        

        for group in self.data['data'][:till]:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    if self.mode == 'test':
                        contexts.append(context)
                        questions.append(question)
                        question_ids.append(qa['question_id'])
                    else: # train or val
                        for ans in qa['answers']:
                            contexts.append(context)
                            questions.append(question)

                            if qa['is_impossible']:
                                answers.append({'text':'','answer_start':-1})
                            else:
                                answers.append(ans)
                
        # return formatted data lists
        return contexts, questions, answers, question_ids
    
    
    def add_end_idx(self, answers, contexts):     # train.json에는 질문에 대한 답이 context 내에서 시작되는 index인 'answer_srart'만 있기 때문에, 추가로 'answer_end'를 찾아주는 함수
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # in case the indices are off 1-2 idxs
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            else:
                for n in [1, 2]:
                    if context[start_idx-n:end_idx-n] == gold_text:
                        answer['answer_start'] = start_idx - n
                        answer['answer_end'] = end_idx - n
                    elif context[start_idx+n:end_idx+n] == gold_text:
                        answer['answer_start'] = start_idx + n
                        answer['answer_end'] = end_idx + n
                        

    def add_token_positions(self, encodings, answers):
        # should use Fast tokenizer
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            if answers[i]['answer_start'] == -1:
                # set [CLS] token as answer if is_impossible
                start_positions.append(0)
                end_positions.append(1)
            else:
                start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))

                assert 'answer_end' in answers[i].keys(), f'no answer_end at {i}'
                end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

            # answer passage truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length                
            # end position cannot be found, shift until found
            shift = 1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
                shift += 1
                
        # char-based -> token based
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})