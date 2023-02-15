# Accuracy Metric Func
import torch 

# Pytorch - Classification에서 Accuracy 보는 함수코드
# train_one_epoch 함수와 valid_one_epoch 함수에 적용할 예정입니다.

def calc_accuracy(X, Y):

    # torch.max를 하면, dim 별로 max 값과 argmax값이 나옵니다. 여기서는 max_indices가 argmax로 나온 index 값입니다. 
    max_vals, max_indices = torch.max(X, 1) 
    
    accuracy = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0] 
    # index 값이 Encoded Target 값(0, 1, 2 , 3)이기에 Target값과 일치할 때, true가 됩니다. 
    # 이를 sum으로 누적시키고 전체 길이로 나눠줍니다. (얼마나 맞았는지 확률이 나오게 됩니다.) 
    
    return accuracy * 100
    # return 할 때 100 곱해서 퍼센트로