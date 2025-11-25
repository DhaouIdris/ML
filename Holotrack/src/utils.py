import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_angle(otf3d: np.ndarray):
    phase = np.angle(np.fft.fftshift(otf3d))
    fig, axes = plt.subplots(1,3, figsize=(19.2,10.8))
    axes = axes.ravel()
    for i in range(3):
        axes[i].imshow(phase[:, :, i], cmap="gray")
    plt.show()