# -*- coding: utf-8 -*-

"""
Filename: traitement_holo.py

Description:
different kind of treatments needeed to hologram analysis or display
Author: Simon BECKER
Date: 2024-07-09

License:
GNU General Public License v3.0

Copyright (C) [2024] Simon BECKER

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
from PIL import Image
import numpy as np
import cupy as cp
import cupy as cp

def read_image(path_image, width = 0, height = 0):
    hologram = np.asarray(Image.open(path_image))

    if ((width != 0) and (height != 0)):

        sx = np.size(hologram, axis = 1)
        sy = np.size(hologram, axis = 0)

        offset_x = (sx - width) // 2
        offset_y = (sy - height) // 2

        hologram = hologram[offset_y:offset_y+height:1, offset_x:offset_x+width:1]
    
    hologram = hologram.astype('float32')
    return hologram

def save_image(img, path_image):
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)

    min_val = img.min()
    max_val = img.max()

    img = ((img - min_val) * 255 / (max_val - min_val)).astype(np.uint8) 
    img = Image.fromarray(img)
    img.save(path_image)

def show_plane(plane):
    if isinstance(plane, cp.ndarray):
        plane = cp.asnumpy(plane)

    min_val = plane.min()
    max_val = plane.max()
    img = Image.fromarray((plane - min_val) * 255 / (max_val - min_val))
    img.show(title="plane")
    img.close()

def compute_module(plane):
    if isinstance(plane, cp.ndarray):
        return(cp.sqrt(cp.square(cp.real(plane)) + cp.square(cp.imag(plane))))
    else:
        return(np.sqrt(np.square(np.real(plane)) + np.square(np.imag(plane))))

def compute_intensity(plane):
    if isinstance(plane, cp.ndarray):
        return(cp.square(cp.real(plane)) + cp.square(cp.imag(plane)))
    else:
        return(np.square(np.real(plane)) + np.square(np.imag(plane)))

def compute_phase(plane):
    if isinstance(plane, cp.ndarray):
        return(cp.arctan(cp.imag(plane) /cp.real(plane)))
    else:
        return(np.arctan(np.imag(plane) /np.real(plane)))

