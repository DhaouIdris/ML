import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from src import utils
from src.data import mb_holonet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True, help=".mat file provided by matlab code")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"The file {args.file} does not exist.")

    mat =  scipy.io.loadmat(args.file)
    data = mat["data"]
    otf3d = mat["otf3d"]
    print(f"{len(data)} holograms loaded.")

    utils.visualize_angle(otf3d)

    idx = np.random.randint(0, data.shape[0])
    hologram = data[idx]

    planes = mb_holonet.MBHolonetHologram.backward_projection(hologram, otf3d)
    cut = planes[:, planes.shape[2]//2, :]
    # plt.scatter(cut[:, 0], cut[:, 1])
    plt.imshow(cut, cmap="gray")
    plt.show()

