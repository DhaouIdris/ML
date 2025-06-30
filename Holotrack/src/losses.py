import torch
import torch.nn as nn

class BinaryCrossEntropySumsMSE(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Custom loss function that combines Cross-Entropy Loss and Mean Squared Error.

        Args:
            alpha (float): Weight for Cross-Entropy Loss.
            beta (float): Weight for Mean Squared Error Loss.
        """
        super(BinaryCrossEntropySumsMSE, self).__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        probs = torch.softmax(logits, dim=1)
        mse_loss = self.mse(probs, targets)
        total_loss = self.alpha * ce_loss + self.beta * mse_loss
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha1, alpha2, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha1 = alpha1  # Weighting factor for the class
        self.alpha2 = alpha2  # Weighting factor for the class
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

        if self.alpha1 is not None and self.alpha2 is not None:
            alpha_factor = targets * self.alpha2 + (1 - targets) * self.alpha1
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
