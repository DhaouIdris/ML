# -*- coding: utf-8 -*-

import os
import numpy as np
import cupy as cp
from argparse import ArgumentParser
import yaml

from src.data import propagation
from src.data import process_hologram
from src.data import hologram
from src.data import otf3d_gen

if __name__ == "__main__":

    #paramètres
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", required=True, )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config["base_dir"]
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    dataset_size = config["dataset_size"]
    ppv_min = config["ppv_min"]
    ppv_max = config["ppv_max"]

    #volume (taille holo & nombre de plans)
    x_size = config["x_size"]
    y_size = config["y_size"]
    z_size = config["z_size"]

    #Phi
    index_milieu = 1.33
    bacterium_index = 1.35

    #Camera
    pixel_size = 5.5e-6
    magnification = 40
    vox_size_xy = pixel_size / magnification
    vox_size_z = 100e-6 / z_size

    #parametres source illumination
    mean = 1.0
    std = 0.
    gaussian_noise = np.abs(np.random.normal(mean, std, [x_size, y_size]))
    
    wavelength = 660e-9
    lambda_medium = wavelength / index_milieu


    positions_path = os.path.join(base_dir, "positions")
    holograms_path = os.path.join(base_dir, "holograms")

    os.makedirs(positions_path, exist_ok=True)
    os.makedirs(holograms_path, exist_ok=True)

    volume_size = [x_size, y_size, z_size]

    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (bacterium_index - index_milieu) / wavelength
    transmission_in_obj = 0.0

    #allocations
    kernel = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)


    # bacteria params
    thickness = 1e-6
    length = 3e-6

    z_list = 5.5e-6 + np.arange(-(z_size-1)//2, (z_size-1)//2 + 1) * 1e-6
    otf3d = otf3d_gen.generate_otf3d_angular({
        'pixel_size': pixel_size,
        'wavelength': wavelength,
        'z_list': z_list,
        'width': x_size,
        'height': y_size,
        "magnification": magnification
    })

    np.save(os.path.join(base_dir,'otf3d.npy'), cp.asnumpy(otf3d))

    for n in range(dataset_size):

        # creation du champs d'illumination
        np_field_plane = np.full(shape=[x_size, y_size], fill_value =  0.0+0.0j, dtype=cp.complex64)
        np_field_plane.real = np.sqrt(gaussian_noise)
        field_plane = cp.asarray(np_field_plane)

        # initialisation du masque (volume 3D booleen présence ou non bactérie)
        mask_volume = np.full(shape = volume_size, fill_value=False, dtype=np.bool8)

        xyz_min_max = np.array([
            [0, x_size * vox_size_xy],
            [0, y_size * vox_size_xy],
            [0, z_size * vox_size_z]
        ])
        
        min_particles = round(ppv_min * x_size*y_size*z_size)
        max_particles = round(ppv_max * x_size*y_size*z_size)
        number_bacteria = np.random.randint(min_particles, max_particles)
        bacteria = hologram.get_random_bacteria(number_bacteria=number_bacteria,
                            xyz_min_max=xyz_min_max, thickness=thickness, length=length)


        for i in range (len(bacteria)):
            # bacteria[i].tofile(positions_path + "/bact_" + str(n) + ".txt")
            hologram.insert_bact_in_mask_volume(mask_volume, bacteria[i], vox_size_xy, vox_size_z)
        
        #invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
        mask_volume =cp.flip(mask_volume, axis=2)


        # SIMU PROPAGATION
        for i in range(mask_volume.shape[2]):
            field_plane = propagation.propagate_angular_spectrum(input_wavefront=field_plane, kernel=kernel,
                                                wavelength=lambda_medium, magnification=magnification, pixel_size=pixel_size, width=x_size, height=y_size, propagation_distance=vox_size_z, min_frequency=0, max_frequency=0)
            
            maskplane = cp.asarray(mask_volume[:,:,i])

            field_plane = hologram.phase_shift_through_plane(mask_plane=maskplane, plane_to_shift=field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj)
            
        process_hologram.save_image(process_hologram.compute_intensity(field_plane), holograms_path + "/holo_" + str(n) + ".bmp")
        img = process_hologram.compute_intensity(field_plane).get()
        with open(os.path.join(positions_path, f"bact_{n}.npy"), "wb") as f:
            np.save(f, mask_volume)

        # process_hologram.show_plane(process_hologram.compute_intensity(field_plane))
        print(f"[{n+1}/{dataset_size}] Range of partciles number: min={min_particles}, max={max_particles}, number of bacteria: {number_bacteria}, number of points: {mask_volume.sum()}")
