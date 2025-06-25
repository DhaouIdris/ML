
# -*- coding: utf-8 -*-

"""
Filename: simu_hologram.py

Description:
Functions needed to generate a virtual volume with objects (spheres and bacteria) includeed in order to create synthetic holograms.
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

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pickle

class Bacterium:
    def __init__(self, x, y, z,
                 thickness, length,
                 theta, phi):
        self.x = x
        self.y = y
        self.z = z
        self.thickness = thickness
        self.length = length
        self.theta = theta
        self.phi = phi

    def tofile(self, path_file):
        data = ["bacterium", self.x, self.y, self.z, self.theta, self.phi]
        with open(path_file, "a") as file:
            file.write("\t".join(map(str, data)) + "\n")

    def coords(self): return np.array([self.x, self.y, self.z])

class Sphere:
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius

    def tofile(self, path_file):
        data = ["sphere", self.x, self.y, self.z, self.radius]
        with open(path_file, "a") as file:
            file.write("\t".join(map(str, data)) + "\n")

    def coords(self): return np.array([self.x, self.y, self.z])

def get_random_bacteria(number_bacteria: int, xyz_min_max: np.ndarray, thickness: float, length: float):
    bacteria = []
    data = {}
    for i,n in enumerate(["x", "y", "z"]):
        data[n] = xyz_min_max[i][0] + (xyz_min_max[i][1] - xyz_min_max[i][0]) * np.random.random(number_bacteria)

    data["theta"] = 90. * np.random.random(number_bacteria)
    data["phi"] = 90. * np.random.random(number_bacteria)

    for i in range(number_bacteria):
        bacteria.append(Bacterium(x=data["x"][i], y=data["y"][i], z=data["z"][i],
                                  thickness=thickness, length=length, theta=data["theta"][i], phi=data["phi"][i]))
    return bacteria

def get_random_spheres(number_spheres: int, xyz_min_max: np.ndarray, radius: float):
    spheres = []
    data = {}
    for i,n in enumerate(["x", "y", "z"]):
        data[n] = xyz_min_max[i][0] + (xyz_min_max[i][1] - xyz_min_max[i][0]) * np.random.random(number_spheres)

    for i in range(number_spheres):
        spheres.append(Sphere(x=data["x"][i], y=data["y"][i], z=data["z"][i], radius=radius))
    return spheres

def plot_bacteria_3d(particles, save_name=None, plot: bool=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # xs = [p.x for p in particles]
    # ys = [p.y for p in particles]
    # zs = [p.z for p in particles]
    # ax.scatter(xs, ys, zs, c='b', marker='o')

    xyz = np.array([p.coords() for p in particles])
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='b', marker='o')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if save_name is not None:
        with open(save_name+".pkl", 'wb') as f:
            pickle.dump(fig, f)
    if plot: plt.show()

def phase_shift_through_plane(mask_plane, plane_to_shift, shift_in_env: float, shift_in_obj: float):
    shift_plane = cp.full(fill_value=shift_in_env, dtype=cp.float32, shape=mask_plane.shape)
    cp.putmask(a=shift_plane, mask=mask_plane, values=shift_in_obj)

    def phase_shift(cplx_plane, shift_plane):
        phase = cp.angle(cplx_plane)
        module = cp.sqrt(cp.real(cplx_plane) ** 2 + cp.imag(cplx_plane) ** 2)
        phase = phase + shift_plane
        return module * cp.exp((0+1.j) * phase)
    
    return phase_shift(plane_to_shift, shift_plane)


def cross_through_plane(mask_plane, plane_to_shift, shift_in_env: float, shift_in_obj: float, transmission_in_obj: float):

    shift_plane = cp.full(fill_value=shift_in_env, dtype=cp.float32, shape=mask_plane.shape)
    transmission_plane = cp.full(fill_value=1.0, dtype=cp.float32, shape=mask_plane.shape)

    cp.putmask(shift_plane, mask=mask_plane, values=shift_in_obj)
    cp.putmask(transmission_plane, mask=mask_plane, values=transmission_in_obj)

    def phase_shift(cplx_plane, shift_plane, transmission_plane):
        phase = cp.angle(cplx_plane)
        module = cp.sqrt(cp.real(cplx_plane) ** 2 + cp.imag(cplx_plane) ** 2)
        phase = phase + shift_plane
        return module* transmission_plane * cp.exp((0+1.j) * phase)
    
    return phase_shift(plane_to_shift, shift_plane, transmission_plane)

def insert_bact_in_mask_volume(mask_volume, bact, vox_size_xy: float, vox_size_z: float):
    bact_pos = bact.coords()
    phi_rad = np.radians(bact.phi)
    theta_rad = np.radians(bact.theta)

    # distance Extremité-centre:
    long_demi_seg = (bact.length - bact.thickness/2.0) / 2.0
    direction = np.array([np.sin(phi_rad) * np.cos(theta_rad), 
                          np.sin(phi_rad) * np.sin(theta_rad), 
                          np.cos(phi_rad)])

	#calcul des positions des extremités du segment interieur de la bactérie m1 et m2
    m1 = bact_pos - long_demi_seg * direction
    m2 = bact_pos + long_demi_seg * direction
    m2m1 = m2-m1 # calcul segment [m2 m1]

    half_length = bact.length / 2.0
    half_thickness = bact.thickness / 2.0
    min_vals = bact_pos - half_length - half_thickness
    max_vals = bact_pos + half_length + half_thickness

    vox_sizes = np.array([vox_size_xy, vox_size_xy, vox_size_z])
    i_min_vals = (min_vals / vox_sizes).astype(int)
    i_max_vals = (max_vals / vox_sizes).astype(int)
    i_min_vals = np.clip(i_min_vals, 0, np.array(mask_volume.shape)-1)
    i_max_vals = np.clip(i_max_vals, 0, np.array(mask_volume.shape)-1)

    # Generate 3D grid
    x_range = np.arange(i_min_vals[0], i_max_vals[0])
    y_range = np.arange(i_min_vals[1], i_max_vals[1])
    z_range = np.arange(i_min_vals[2], i_max_vals[2])
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    pos_x = x_grid * vox_size_xy
    pos_y = y_grid * vox_size_xy
    pos_z = z_grid * vox_size_z

    vox_m1 = np.stack([pos_x - m1[0], pos_y - m1[1], pos_z - m1[2]], axis=-1)
    m2m1_norm = np.linalg.norm(m2m1)
    # calcul de la distance de la position xyz avec le segment [m1 m2]
    distance = np.linalg.norm(np.cross(m2m1, vox_m1), axis=-1) / m2m1_norm
    mask_volume[x_grid, y_grid, z_grid] = distance < bact.thickness/2

def insert_sphere_in_mask_volume(mask_volume, sphere, vox_size_xy: float, vox_size_z: float):
    sphere_pos = sphere.coords()

    #calcul de la box autour de la sphere
    min_vals = sphere_pos - sphere.radius
    max_vals = sphere_pos + sphere.radius

    vox_sizes = np.array([vox_size_xy, vox_size_xy, vox_size_z])
    i_min_vals = (min_vals / vox_sizes).astype(int)
    i_max_vals = (max_vals / vox_sizes).astype(int)
    i_min_vals = np.clip(i_min_vals, 0, np.array(mask_volume.shape)-1)
    i_max_vals = np.clip(i_max_vals, 0, np.array(mask_volume.shape)-1)

    # Generate 3D grid
    x_range = np.arange(i_min_vals[0], i_max_vals[0])
    y_range = np.arange(i_min_vals[1], i_max_vals[1])
    z_range = np.arange(i_min_vals[2], i_max_vals[2])
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    pos_x = x_grid * vox_size_xy
    pos_y = y_grid * vox_size_xy
    pos_z = z_grid * vox_size_z

    vox_pos = np.stack([pos_x, pos_y, pos_z], axis=-1)

    distance = np.sqrt(np.sum((vox_pos-sphere_pos)**2, axis=-1))
    mask_volume[x_grid, y_grid, z_grid] = distance < sphere.radius

if __name__ == "__main__":
    plot_bacteria_3d(get_random_bacteria(1000, np.array([[-1,1],[-1,1],[-1,1]]), 0.1, 0.1), plot=True)
