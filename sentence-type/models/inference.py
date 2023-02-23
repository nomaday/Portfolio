import gc

from configs.config import config
from utils.preprocessing import *
from models.model import *
from models.trainer import *
from train import *


# Better F1 Score Model paths
model_paths_f1 = []

for num in range(0, config['n_folds']):
    model_paths_f1.append(model_save + f"Loss-Fold-{num}_f1.bin")

print(len(model_paths_f1))

# Better Loss Model paths
model_paths = []

for num in range(0, config['n_folds']):
    model_paths.append(model_save + f"Loss-Fold-{num}.bin")

print(len(model_paths))


def inference(model_paths, dataloader, device=config['device']):

    final_type_preds, final_pn_preds, final_time_preds, final_sure_preds = [], [], [], []
    
    for i, path in enumerate(model_paths):
        model = Model(config['model'])
        model.to(config['device'])
        model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        preds = test_func(model, dataloader, device)

        final_type_preds.append(preds['type'])
        final_pn_preds.append(preds['pn'])
        final_time_preds.append(preds['time'])
        final_sure_preds.append(preds['sure'])
        
    # TYPE: 유형
    final_type_preds = np.array(final_type_preds)
    final_type_preds = np.mean(final_type_preds, axis = 0)

    # PN: 극성
    final_pn_preds = np.array(final_pn_preds)
    final_pn_preds = np.mean(final_pn_preds, axis = 0)

    # TIME: 시제
    final_time_preds = np.array(final_time_preds)
    final_time_preds = np.mean(final_time_preds, axis = 0)

    # SURE: 확실성
    final_sure_preds = np.array(final_sure_preds)
    final_sure_preds = np.mean(final_sure_preds, axis = 0)

    final_preds = dict()
    final_preds['type'] = final_type_preds
    final_preds['pn'] = final_pn_preds
    final_preds['time'] = final_time_preds
    final_preds['sure'] = final_sure_preds
    
    gc.collect()

    return final_preds


# f1 Inference Argmax
f1_preds = inference(model_paths_f1, test_loader, config['device'])

ss = pd.read_csv(data_path + "sample_submission.csv")
print(ss.shape)

# target inverse encoders
target4_inverse = {True: '확실', False: '불확실'}

inverse_encode = {'type': target1_inverse, 
                  'pn': target2_inverse, 
                  'time': target3_inverse, 
                  'sure': target4_inverse, }


def column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'type', inverse_encode = inverse_encode, threshold = .5):

    print("column_name = 'time: ", column)
 
    if column == 'sure':
        class_index = f1_preds[column] > threshold
    else:
        class_index = np.argmax(f1_preds[column], axis = 1)
        print("Shape of preds: ", class_index.shape)

    print()
    print(ss.shape)
    ss[column] = class_index
    print(ss[column].value_counts())
    print()
    ss[column] = ss[column].apply(lambda x: inverse_encode[column][x])
    print(ss.shape)
    print(ss[column].value_counts())
    print()


column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'type', )
column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'pn',)
column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'time')
column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'sure')


ss.label = 0

for index in range(0, len(ss)):
    ss.loc[index, 'label'] = ss.loc[index, 'type'] + '-' + ss.loc[index, 'pn'] + '-' + ss.loc[index, 'time'] + '-' + ss.loc[index, 'sure']

ss.drop(['type', 'pn', 'time', 'sure'], axis = 1, inplace = True)


# ## Submission CSV file Save
print(sub_path)

trans = str.maketrans('/', '-')
config['model'].translate(trans)

ss.to_csv(f"{config['testnum']}_{config['model'].translate(trans)}_bs_{config['train_batch_size']}_epoch_{config['n_epochs']}_folds_{config['n_folds']}_lr_{config['learning_rate']}.csv", index=False) 