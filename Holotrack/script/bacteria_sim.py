# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import cupy as cp

from src.data import propagation
from src.data import process_hologram
from src.data import hologram

if __name__ == "__main__":
   
    # paramètres
    base_dir = "simu_holo_bact"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # volume (taille holo & nombre de plans)
    x_size = 1024
    y_size = 1024
    z_size = 200

    # Phi
    # transmission_milieu = 1.0
    # transmission_bacterie = 1.0
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
    # transmission = 1.0

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
        [900*vox_size_xy, 900*vox_size_xy, 100*vox_size_z, 0.0, 80.0],
        [1000*vox_size_xy, 1000*vox_size_xy, 100*vox_size_z, 0.0, 90.0]
    ]

    # source illumination parameters

    mean = 1.0
    std = 0.1
    gaussian_noise = np.abs(np.random.normal(mean, std, [x_size, y_size]))
    
    wavelength = 660e-9
    lambda_medium = wavelength / index_medium

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date and hour", formatted_date_time)

    positions_path = os.path.join(base_dir,formatted_date_time, "positions")
    holograms_path = os.path.join(base_dir,formatted_date_time, "holograms")

    os.makedirs(positions_path, exist_ok=True)
    os.makedirs(holograms_path, exist_ok=True)

    volume_size = [x_size, y_size, z_size]

    # Bacteria
    bacteria = []
    for b in positions_bact:
        bact = hologram.Bacterium(
            x=b[0], y=b[1], z=b[2], thickness=thickness, length=length, theta=b[3], phi=b[4])
        bacteria.append(bact)
        bact.tofile(os.path.join(positions_path, "bact_positions.txt"))

    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (bacterium_index - index_medium) / wavelength

    # Allocations
    kernel = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)

    # creation du champs d'illumination
    np_field_plane = np.full(shape=[x_size, y_size], fill_value =  0.0+0.0j, dtype=cp.complex64)
    np_field_plane.real = np.sqrt(gaussian_noise)
    field_plane = cp.asarray(np_field_plane)

    # initialisation du masque (volume 3D booleen présence ou non bactérie)
    mask_volume = np.full(shape = volume_size, fill_value=False)

    # insertion des bactéries dans le volume
    for i in range (len(bacteria)):
        hologram.insert_bact_in_mask_volume(mask_volume, bacteria[i], vox_size_xy, vox_size_z)
        print("bact " + str(i) + " ok")
        
    # invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
    mask_volume =cp.flip(mask_volume, axis=2)


    # SIMU PROPAGATION
    for i in range(mask_volume.shape[2]):
            
        field_plane = propagation.propagate_angular_spectrum(input_wavefront=field_plane, kernel=kernel,
                                                wavelength=lambda_medium, magnification=magnification, pixel_size=pixel_size, width=x_size, height=y_size, propagation_distance=vox_size_z, min_frequency=0, max_frequency=0)
            
        maskplane = cp.asarray(mask_volume[:,:,i])

        field_plane = hologram.phase_shift_through_plane(mask_plane=maskplane, plane_to_shift=field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj)
            
    process_hologram.save_image(process_hologram.compute_intensity(field_plane), holograms_path + "/holo_simu.bmp")

    process_hologram.show_plane(process_hologram.compute_intensity(field_plane))

