import os
import pandas as pd

from configs.config import *
from models.model import *
from train import *
from utils.preprocessing import *


model = electra(pretrained="monologg/koelectra-base-v3-discriminator").to(device)

checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))

model.load_state_dict(checkpoint['model'])

model.eval()

pred_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

for batch_index, batch in enumerate(tqdm(test_dataloader, leave=True)):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Inference
    outputs = model(input_ids, attention_mask=attention_mask)

    start_score = outputs.start_logits
    end_score = outputs.end_logits

    start_idx = torch.argmax(start_score, dim=1).cpu().tolist()
    end_idx = torch.argmax(end_score, dim=1).cpu().tolist()

    y_pred = []
    for i in range(len(input_ids)):
        if start_idx[i] > end_idx[i]:
            output = ''

        ans_txt = tokenizer.decode(input_ids[i][start_idx[i]:end_idx[i]]).replace('#','')

        if ans_txt == '[CLS]':
            ans_txt == ''

        y_pred.append(ans_txt)


    q_end_idx = BATCH_SIZE*batch_index + len(y_pred)
    for q_id, pred in zip(question_ids[BATCH_SIZE*batch_index:q_end_idx], y_pred):
        pred_df.loc[pred_df['question_id'] == q_id,'answer_text'] = pred


# Set predict serial
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = predict_timestamp
predict_serial

PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

pred_df.to_csv(os.path.join(PREDICT_DIR, f'prediction_lr{LEARNING_RATE}_bs{BATCH_SIZE}_epoch{EPOCHS}_split0.2_koelectra-base.csv'), index=False)