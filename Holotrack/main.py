#write train.py to train the model
#         pt = torch.exp(-BCE_loss)  # Probability of the true class
#         loss = self.alpha1 * (1 - pt) ** self.gamma * BCE_loss
import os
import datetime
import numpy as np
import cupy as cp

from src.data import propagation
from src.data import process_hologram
from src.data import hologram

from src.models import model
from src.losses import BinaryCrossEntropySumsMSE, FocalLoss
from src.utils import train_utils

from src.utils import train_utils as utils
from src.utils import train_utils as train_utils

from src.utils import train_utils as train_utils


if __name__ == "__main__":
    # Parameters
    base_dir = "simu_holo_bact"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Volume (hologram size & number of planes)
    x_size = 1024
    y_size = 1024
    z_size = 200

    # Phi
    index_medium = 1.33
    bacterium_index = 1.35

    # Camera
    pixel_size = 5.5e-6
    magnification = 40
    vox_size_xy = pixel_size / magnification
    vox_size_z = 100e-6 / z_size

    # Bacteria
    thickness = 1e-6
    length = 3e-6

    angles = [0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    positions_bact = [
        [100*vox_size_xy, 100*vox_size_xy, 100*vox_size_z, 0.0, 0.0],
        [200*vox_size_xy, 200*vox_size_xy, 100*vox_size_z, 0.0, 10.0],
        [300*vox_size_xy, 300*vox_size_xy, 100*vox_size_z, 0.0, 20.0],
        [400*vox_size_xy, 400*vox_size_xy, 100*vox_size_z, 0.0, 30.0],
        [500*vox_size_xy, 500*vox_size_xy, 100*vox_size_z, 0.0, 40.0],
        [600*vox_size_xy, 600*vox_size_xy, 100*vox_size_z, 0.0, 50.0],
        [700*vox_size_xy, 700*vox_size_xy, 100*vox_size_z, 0.0, 60.0],
        [800*vox_size_xy, 800*vox_size_xy, 100*vox_size_z, 0.0, 70.0],
        [900*vox_size_xy, 900*vox_size_xy, 100*vox
_size_z, 0.0, 80.0],
        [1000*vox_size_xy, 1000*vox_size_xy, 100*vox_size_z, 0.0, 90.0]