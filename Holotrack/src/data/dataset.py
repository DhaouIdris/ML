
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

class BacteriaDataset(torch.utils.data.Dataset):
    def __init__(self, root, valid_ratio=0.2, train_batch_size=8, eval_batch_size=8, num_workers=4):
        self.root = root
        self.valid_ratio = valid_ratio
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.holograms = [entry.path for entry in os.scandir(os.path.join(root, "holograms"))]
        self.positions = [entry.path for entry in os.scandir(os.path.join(root, "positions"))]
        # TODO: change | propose to use same id for holo and pos
        self.holograms = sorted(self.holograms)
        self.positions = sorted(self.positions)
        self.otf_fname = os.path.join(root, "otf3d.npy")
        self.params = {}

    def __len__(self,): return len(self.holograms)

    def __str__(self): return f"BacteriaDataset(root={self.root}, valid_ratio={self.valid_ratio}, train_batch_size={self.train_batch_size}, eval_batch_size={self.eval_batch_size}, num_workers={self.num_workers})"

    def __repr__(self) -> str: return self.__str__()

    def split(self):
        indices = list(range(len(self)))
        random.shuffle(indices)
        num_valid = int(self.valid_ratio * len(self))
        self.train_indices = indices[num_valid:]
        self.valid_indices = indices[:num_valid]
        self.train = torch.utils.data.Subset(self, self.train_indices)
        self.eval = torch.utils.data.Subset(self, self.valid_indices)

    def get_dataloader(self, train: bool):
        data = self.train if train else self.eval
        batch_size = self.train_batch_size if train else self.eval_batch_size
        
        return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size,
                                           shuffle=train, num_workers=self.num_workers)

    def train_dataloader(self): return self.get_dataloader(train=True)
    def eval_dataloader(self): return self.get_dataloader(train=False)

    def __getitem__(self, idx):
        img_fname = self.holograms[idx]
        pos_fname = self.positions[idx]

        positions = self.get_positions(pos_fname)
        img = self.get_hologram(img_fname)

        return img.astype(np.float32), positions.astype(np.float32)

    def get_otf(self):
        with open(self.otf_fname, "rb") as f:
            otf = np.load(f)
        return otf


    def get_positions(self, fname):
        with open(fname, "rb") as f:
            return np.load(f)
    # def get_positions(self, fname):
    #     with open(fname, 'r') as f:
    #         positions = f.readlines()
    #
    #         positions = list(map(lambda x: x.split(), positions))
    #         positions = [list(map(float, x)) for x in positions]
    #     return positions

    def get_hologram(self, fname):
        with Image.open(fname) as img:
            return np.array(img)

    def plot_3d(self, positions, ax=None, color=None, label=None, marker='o'):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = np.where(positions == 1)
        ax.scatter(xs, ys, zs, c=color, marker=marker, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def plot_2d(self, positions, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        xs, ys, _ = np.where(positions == 1)
        ax.scatter(xs, ys, c='b', marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

class MBHoloDataset(BacteriaDataset):
    def __init__(self, root, valid_ratio=0.2, train_batch_size=8, eval_batch_size=8, num_workers=4):
        super(MBHoloDataset, self).__init__(root=root, valid_ratio=valid_ratio, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, num_workers=num_workers)
        self.otf_fname = os.path.join(root, "otf3d.npy")
        if os.path.exists(os.path.join(root, "params.json")):
            with open(os.path.join(root, "params.json"), "r") as f:
                self.params = json.load(f)
                self.params["z_range"] = "-".join([str(v) for v in self.params["z_range"]])
        else:
            self.params = {}

    def __getitem__(self, idx):
        img_fname = self.holograms[idx]
        pos_fname = self.positions[idx]
        return self.get_hologram(img_fname).astype(np.float32), self.get_positions(pos_fname).astype(np.float32)

    def get_positions(self, fname):
        with open(fname, "rb") as f:
            return np.load(f)

    def get_hologram(self, fname):
        with open(fname, "rb") as f:
            return np.load(f)


    def __str__(self): return f"MBHoloDataset(root={self.root}, valid_ratio={self.valid_ratio}, train_batch_size={self.train_batch_size}, eval_batch_size={self.eval_batch_size}, num_workers={self.num_workers}, {str(self.params)})"

