import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        def forward(self, inputs, targets):
            
            # Compute the cross-entropy loss
            ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
            # Compute the focal loss
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
            # Apply reduction
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss