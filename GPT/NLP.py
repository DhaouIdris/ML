import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocabulary = set()
        for sentance in positive:
            for word in sentance.split():
                vocabulary.add(word)
        for sentance in negative:
            for word in sentance.split():
                vocabulary.add(word)

        sorted_list = sorted(list(vocabulary)
        str_to_int = {}
        for i in range(len(sorted_list)):
            str_to_int[sorted_list[i]] = i + 1

        tensors = []
        
        pass
