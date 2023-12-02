import torch
import torch.nn as nn
import torchvision
import torch.utils.data



class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __followitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), img

    def __len__(self):
        return len(self.base_dataset)
