import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def show_plankton_image(img, mask):
    """
    Display an image and its mask side by side

    img is either (H, W, 1)
    mask is either (H, W)
    """

    img = img.squeeze()

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, interpolation="none", cmap="tab20c")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("plankton_sample.png", bbox_inches="tight", dpi=300)
    plt.show()
