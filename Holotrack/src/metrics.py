import torch

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def precision(y_true, y_pred):
    numerator = (y_true*y_pred).sum()
    denominator = y_pred.sum()
    return numerator/denominator.clamp(min=1e-6)

def recall(y_true, y_pred):
    numerator = (y_true*y_pred).sum()
    denominator = y_true.sum()
    return numerator/denominator.clamp(min=1e-6)

def f1_score(y_true, y_pred):
    numerator = 2 * (y_true*y_pred).sum()
    denominator = (y_true.sum() + y_pred.sum())
    return numerator/denominator if denominator>0 else torch.tensor(0.0)

