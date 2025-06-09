"""
src/data/otf3d_gen.py

Optical Transfer Function (OTF) Generation Module

This module provides GPU-accelerated methods for calculating 3D Optical Transfer Functions (OTF3D)
using different optical propagation models. Contains implementations for both Fresnel approximation
and Angular Spectrum methods.

Functions:
----------
generate_otf3d(params):
    Computes OTF using Fresnel approximation with Taylor series expansion.
    Suitable for moderate numerical apertures and propagation distances.

generate_otf3d_angular(params):
    GPU-accelerated OTF calculation using Angular Spectrum method with CuPy.
    Provides more accurate results for high numerical aperture systems.

Key Features:
------------
- Supports both CPU (NumPy) and GPU (CuPy) implementations
- Implements different optical propagation models
- Handles complex wavefront calculations
- Optimized memory management for 3D OTF arrays

Parameters:
----------
All functions expect a params dictionary containing:
- pixel_size: Sensor pixel size in meters
- wavelength: Light wavelength in meters
- z_list: Array/list of propagation distances
- height/width: OTF dimensions in pixels
- magnification: Optical system magnification (for angular spectrum method)

"""
import numpy as np
import cupy as cp
import matplotlib as plt
from src.data.propagation import calculate_angular_spectrum_propagation_kernel

def generate_otf3d(params):
    pph = params['pixel_size']
    lambda_ = params['wavelength']
    z_list = params['z_list']
    Ny = params['height']
    Nx = params['width']

    # Constant frequencies
    # Write it this way to avoid fftshifts
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    fx = (np.mod(X + Nx / 2, Nx) - np.floor(Nx / 2)) / Nx
    fy = (np.mod(Y + Ny / 2, Ny) - np.floor(Ny / 2)) / Ny

    term = (fx**2 + fy**2) * (lambda_ / pph)**2

    # Fresnel expansion: (1+x)^(1/2) = 1 + a * x + a*(a-1)/2! * x^2
    final_term = -1/2 * term - 1/8 * term**2 - 1/16 * term**3

    # Make sure the sign is correct
    otf3d = [np.exp(1j * 2 * np.pi / lambda_ * z * final_term) for z in z_list]
    otf3d = np.stack(otf3d, axis=2)

    return otf3d

def generate_otf3d_angular(params):
    pixel_size = params['pixel_size']
    wavelength = params['wavelength']
    magnification = params['magnification']
    width = params['width']
    height = params['height']
    z_list = params['z_list']
    width = cp.int32(width)
    height = cp.int32(height)
    otf3d = cp.zeros((height, width, len(z_list)), dtype=cp.complex64)
    for i,z in enumerate(z_list):
        kernel = cp.zeros((height,width),dtype = cp.complex64)
        threads_per_block = 256  # Ajuster
        blocks_per_grid = (width * height + threads_per_block - 1) // threads_per_block

        calculate_angular_spectrum_propagation_kernel[blocks_per_grid, threads_per_block](
            kernel, wavelength, magnification, pixel_size, width, height, z
        )
        kernel = np.roll(kernel, shift=(kernel.shape[0] // 2, kernel.shape[1] // 2), axis=(0, 1))
        otf3d[:,:,i] = kernel
    return otf3d


if __name__ == '__main__':
    params_fresnel = {
    'pixel_size': 5.5e-6/40,         # Pas de pixel de la caméra CCD
    'wavelength': 660e-9,     # Longueur d'onde de la lumière
    'z_list': 5.5e-6 + np.arange(-(256-1)//2, (256-1)//2 + 1) * 1e-6,  # Distances de propagation
    'height': 256,             # Hauteur de l'OTF
    'width': 256,              # Largeur de l'OTF
    } 
    params_MB = {
    'pixel_size': 20e-6,         # Pas de pixel de la caméra CCD
    'wavelength': 632e-9,     # Longueur d'onde de la lumière
    'z_list': [10e-3 + i * 50e-6 for i in range(128)],  # Distances de propagation
    'height': 128,             # Hauteur de l'OTF
    'width': 128,              # Largeur de l'OTF
    'magnification':1
    }
    params = {
    'pixel_size': 5.5e-6,       # Taille du pixel de la caméra
    'wavelength': 660e-9,  # Longueur d'onde ajustée pour l'indice de réfraction
    'z_list': 5.5e-6 + np.arange(-(256-1)//2, (256-1)//2 + 1) * 1e-6,  # Distances
    'height': 256,              # Hauteur
    'width': 256,               # Largeur
    'magnification': 40        # Grossissement correct
}
    #otf3d=generate_otf3d(params_fresnel)
    otf3d = generate_otf3d_angular(params)
    np.save('otf3d_angular.npy', cp.asnumpy(otf3d))
 