
import torch

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

def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return intersection / union.clamp(min=1e-6)

def compute_metrics(y_true, y_pred):
    return {
        "precision": precision(y_true=y_true, y_pred=y_pred).item(),
        "recall": recall(y_true=y_true, y_pred=y_pred).item(),
        "f1": f1_score(y_true=y_true, y_pred=y_pred).item(),
        "iou": iou(y_true, y_pred).item()
    }
