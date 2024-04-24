import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        pass

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        pass

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        pass

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        pass
