import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List, Tuple

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocabulary = set()
        for sentence in positive:
            for word in sentance.split():
                vocabulary.add(word)
        for sentence in negative:
            for word in sentance.split():
                vocabulary.add(word)

        sorted_list = sorted(list(vocabulary)
        str_to_int = {}
        for i in range(len(sorted_list)):
            str_to_int[sorted_list[i]] = i + 1

        tensors_list = []

        for sentence in positive:
            list = []
            for word in sentence.split():
                list.append(str_to_int[word])
            tensors_list.append(torch.tensor(list))


        for sentence in negative:
            list = []
            for word in sentence.split():
                list.append(str_to_int[word])
            tensors_list.append(torch.tensor(list))

        return nn.utils.rnn.pad_sequence(tensors, batch_first = True




    #Dataset_loader



class Solution:
    def batch_loader(self, raw_dataset, context_length, batch_size):
        torch.manual_seed(0)
        list_words = raw_dataset.split()
        index = torch.randint(low=0, high= len(list_words) - context_length, size = (batch_size,))
        X= []
        Y= []
        
                        

