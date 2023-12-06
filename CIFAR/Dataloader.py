import os

import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from torchvision import transforms




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


def get_dataloaders(config, use_cuda):
    """
        Arguments :
        ratio       : float in ]0,1[   , pourcentage de donnees de train qui vont etre utilisees pour la validation
        batch_size  : integer          , taille des batchs
        num_workers : integer          , nombre de processus paralleles pour charger les donnees
        use_cuda    : Boolean          , Utilisation d'un gpu
        data_dir    :
    """

    ratio = config["data"]["valid_ratio"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    data_dir = config["data"]["trainpath"]

    if config["data"]["dataset"] in ["CIFAR10", "CIFAR100"]:
        dataset = eval(
            f"torchvision.datasets.{config['data']['dataset']}(root=data_dir, train=True, download=True)"
        )
    else:
        raise ValueError("Can only process CIFAR10 or CIFAR100")

    classes = dataset.classes
    num_classes = len(classes)

    train_size = int((1 - ratio) * len(dataset))
    valid_size = int(ratio * len(dataset))

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    # Creation d'un dataloader intermediaire afin de calculer la moyenne et l'ecart type de l'ensemble d'entrainement

    normalizing_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    normalizing_loader = torch.utils.data.DataLoader(
        dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Compute mean and variance from the training set
    mean, std = utils.mean_std(normalizing_loader)


        # Data Augmentation and Normalization
    train_augmentation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    test_normalization = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Application des transformations
    train_dataset = DatasetTransformer(train_dataset, train_augmentation)
    valid_dataset = DatasetTransformer(valid_dataset, test_normalization)

    a = train_dataset.__followitem__(0)

    # Creation dataloader

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    # test_loader  = torch.utils.data.DataLoader(test_dataset , batch_size=batch_size, shuffle=False, num_workers=num_workers)

    X, _ = train_dataset[0]
    input_size = tuple(X.shape)  # C, H, W

    return train_loader, valid_loader, input_size, num_classes, classes, a, mean, std

