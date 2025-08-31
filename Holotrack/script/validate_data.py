import numpy as np
import matplotlib.pyplot as plt
import imageio
from argparse import ArgumentParser
import os

from src.data import mb_holonet
from src.data import dataset

if __name__ == "__main__":
    os.makedirs("data-validation", exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--data", "-d", required=True, type=str, help="dataset")
    parser.add_argument("--indice", "-i", required=True, type=int, help="indice of dataset")
    args = parser.parse_args()


    data = dataset.MBHoloDataset(root=args.data)

    with open(data.otf_fname, "rb") as f: 
        otf3d = np.load(f)

    data_shape = 'x'.join([str(l) for l in otf3d.shape])
    img, _, ground_truth = data[args.indice]
    vol =  mb_holonet.MBHolonetHologram.backward_projection(img, otf3d)

    save_dir = f"data-validation/{data_shape}-{args.indice}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/superposition", exist_ok=True)
    os.makedirs(f"{save_dir}/coupe-xz", exist_ok=True)
    plt.imshow(img, cmap="gray")
    plt.title("hologram")
    plt.savefig(f"{save_dir}/0-hologram.png")
    superposition_imgs = []
    for i in range(vol.shape[2]):
        plt.imshow(vol[:, :, i], cmap="gray")
        plt.title(f"superposition: z={i}")
        gt = ground_truth[:, :, i]
        y_coords, x_coords = np.where(gt > 0)
        plt.scatter(x_coords, y_coords, color="red", marker="o", s=5)
        plt.savefig(f"{save_dir}/superposition/z={i}.png")
        superposition_imgs.append(imageio.imread(f"{save_dir}/superposition/z={i}.png"))
        plt.close()

    video_path = f"{save_dir}/superposition.gif"
    imageio.mimsave(video_path, superposition_imgs, fps=5)
    print(f"Video saved at: {video_path}")

    cut_imgs = []
    for i in range(vol.shape[1]):
        plt.imshow(vol[:, i, :], cmap="gray")
        plt.title(f"coupe x/z: y={i}")
        gt = ground_truth[:, i, :]
        z_coords, x_coords = np.where(gt > 0)
        plt.scatter(x_coords, z_coords, color="red", marker="o", s=5)
        plt.savefig(f"{save_dir}/coupe-xz/y={i}.png")
        cut_imgs.append(imageio.imread(f"{save_dir}/coupe-xz/y={i}.png"))
        plt.close()

    video_path = f"{save_dir}/coupe-xz.gif"
    imageio.mimsave(video_path, cut_imgs, fps=5)
    print(f"Video saved at: {video_path}")
