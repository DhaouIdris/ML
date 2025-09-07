import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from src import data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", "-d", required=True, type=str, help="Dataset dir. Ex: data/mb-holonet-dataset-float64-0", default="data/mb-holonet-dataset-float64-0")
    parser.add_argument("--type", "-t", type=str, required=True, default="bacteria", help="Dataset type: bacteria or mbholonet")
    args = parser.parse_args()

    if args.type=="mbholonet":
        dataset = data.MBHoloDataset(root=args.data)
    elif args.type=="bacteria":
        dataset = data.BacteriaDataset(root=args.data)
    else:
        print(f"{args.type} is unknown. Use bacteria or mbholonet")
    img, pos = dataset[np.random.randint(len(dataset))]


    fig = plt.figure(figsize=(14, 6))

    ax_img = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

    ax_img.imshow(img, cmap='gray')
    ax_img.set_title("hologram")
    ax_img.axis("off")

    dataset.plot_3d(pos, ax=ax_3d)
    ax_3d.set_title("gt: volume")

    plt.show()

