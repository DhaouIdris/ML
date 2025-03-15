# coding: utf-8

# External imports
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for the class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):

        # Calculate the cross entropy loss
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        # Apply sigmoid to logits to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Calculate p_t
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_factor = targets * self.alpha[1] + (1 - targets) * self.alpha[0]
            focal_weight = alpha_factor * focal_weight

        # Apply focal weight to loss
        loss = focal_weight * BCE_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Return per-element loss if no reduction

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to logits
        inputs = torch.sigmoid(inputs)

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def get_loss(lossname):
    if lossname == "FocalLoss":
        alpha = config["loss"]["params"]["alpha"]
        gamma = config["loss"]["params"]["gamma"]
        return lossf.FocalLoss(alpha=alpha, gamma=gamma)
    return eval(f"nn.{lossname}()")



def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
