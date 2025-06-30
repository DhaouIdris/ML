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
