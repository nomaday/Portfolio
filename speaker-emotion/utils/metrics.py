from sklearn.metrics import f1_score

 
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")