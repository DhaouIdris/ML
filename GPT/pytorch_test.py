import torch
import torch.nn
from torchtyping import TensorType

class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        L = list(to_reshape.size())
        return torch.reshape(to_reshape, (L[0]*L[1]//2, 2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        return torch.mean(to_avg, 0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        return torch.cat((cat_one, cat_two), 1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        return torch.nn.functional.mse_loss(prediction, target)


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        pass
        # Define the architecture here
        first_layer = nn.Linear(28*28, 512)
        second_layer = nn.Dropout(0,2)
        final_layer = nn.Linear(512, 10)
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        pass
        # Return the model's prediction to 4 decimal places
        return self.final_layer(self.second_layer(nn.functional.relu(self.first_layer(images))))
