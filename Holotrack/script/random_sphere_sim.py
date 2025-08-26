# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import cupy as cp

from src.data import propagation
from src.data import process_hologram
from src.data import hologram

if __name__ == "__main__":
   
    #paramètres
    base_dir = "simu_holo"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    #volume (taille holo & nombre de plans)
    x_size = 1024
    y_size = 1024
    z_size = 200

    #Phi
    transmission_milieu = 1.0
    transmission_sphere = 1.0
    index_medium = 1.33
    index_sphere = 1.35

    #Camera
    pixel_size = 5.5e-6
    magnification = 40
    vox_size_xy = pixel_size / magnification
    vox_size_z = 100e-6 / z_size

    #liste sphere
    radius = 0.8e-6
    transmission = 0.0

    positions_spheres = [
        [256*vox_size_xy, 256*vox_size_xy, 50*vox_size_z, radius],
        [512*vox_size_xy, 512*vox_size_xy, 100*vox_size_z, radius],
        [768*vox_size_xy, 768*vox_size_xy, 150*vox_size_z, radius]
    ]

    #parametres source illumination
    mean = 1.0
    std = 0.01
    gaussian_noise = np.abs(np.random.normal(mean, std, [x_size, y_size]))
    
    wavelenght = 660e-9
    lambda_medium = wavelenght / index_medium


    #Creation des repertoires d'enregistrement:
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date et heure actuelles:", formatted_date_time)

    positions_path = os.path.join(base_dir,formatted_date_time, "positions")
    holograms_path = os.path.join(base_dir,formatted_date_time, "holograms")

    os.makedirs(positions_path, exist_ok=True)
    os.makedirs(holograms_path, exist_ok=True)

    volume_size = [x_size, y_size, z_size]

    #creation des bactéries
    spheres = []
    for s in positions_spheres:
        sph = hologram.Sphere(x=s[0], y=s[1], z=s[2], radius=s[3])
        spheres.append(sph)
        sph.tofile(positions_path + "/spheres_positions.txt")

    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_sphere - index_medium) / wavelenght

    #allocations
    h_holo = np.zeros(shape = (x_size, y_size), dtype = np.float32)
    d_holo = cp.zeros(shape = (x_size, y_size), dtype = cp.float32)
    d_fft_holo = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)
    d_fft_holo_propag = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)
    d_holo_propag = cp.zeros(shape = (x_size, y_size), dtype = cp.float32)
    kernel = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)

    #creation du champs d'illumination
    np_field_plane = np.full(shape=[x_size, y_size], fill_value =  0.0+0.0j, dtype=cp.complex64)
    np_field_plane.real = np.sqrt(gaussian_noise)
    field_plane = cp.asarray(np_field_plane)

    #initialisation du masque (volume 3D booleen présence ou non bactérie)
    mask_volume = np.full(shape = volume_size, fill_value=False, dtype=np.bool8)

    #lecture fichier positions

    #insertion des bactéries dans le volume
    for i in range (len(spheres)):
        hologram.insert_sphere_in_mask_volume(mask_volume, spheres[i], vox_size_xy, vox_size_z)
        print("bact " + str(i) + " ok")
        
    #invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
    mask_volume =cp.flip(mask_volume, axis=2)


        #SIMU PROPAGATION
    for i in range(mask_volume.shape[2]):
        field_plane = propagation.propagate_angular_spectrum(
            input_wavefront=field_plane, kernel=kernel,
            wavelength=lambda_medium, magnification=magnification,
            pixel_size=pixel_size, width=x_size, height=y_size,
            propagation_distance=vox_size_z, min_frequency=0, max_frequency=0)
            
        maskplane = cp.asarray(mask_volume[:,:,i])

        field_plane = hologram.cross_through_plane(mask_plane=maskplane, plane_to_shift=field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj, transmission_in_obj=transmission_sphere)
            
    process_hologram.save_image(process_hologram.compute_intensity(field_plane), holograms_path + "/holo_simu.bmp")

    process_hologram.show_plane(process_hologram.compute_intensity(field_plane))
