## -*- coding: utf-8 -*-

"""
Filename: propagation.py

Description:
Groups of functions needed for retro-propagates holograms (Fresnell, Rayleigh-Sommerfeld and angular spectrum).
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

import math
import cupy as cp
from cupyx import jit


#########################################################################################################################################
################################               calculs des KERNELS                  #####################################################
#########################################################################################################################################

@jit.rawkernel()
def apply_bandpass_filter(input_fft_plane: cp.ndarray, output_fft_plane: cp.ndarray, width: int, height: int, min_freq: float, max_freq: float) -> None:
    """
    Applies a bandpass filter to the input FFT plane, retaining frequencies between min_freq_px and max_freq_px.

    Args:
        input_fft_plane (cp.ndarray): Input FFT plane (complex data) on GPU, representing the frequency domain.
        output_fft_plane (cp.ndarray): Output FFT plane (complex data) on GPU, where the filtered result will be stored.
        width (int): Width of the FFT planes (in pixels).
        height (int): Height of the FFT planes (in pixels).
        min_freq_px (float): Minimum frequency (in pixels) to retain in the bandpass filter.
        max_freq_px (float): Maximum frequency (in pixels) to retain in the bandpass filter.

    Notes:
        - The filter retains frequencies where min_freq_px < F < max_freq_px.
        - Both input_fft_plane and output_fft_plane are complex arrays of size (height, width).
    """

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    jj = cp.int32(index) // cp.int32(width)
    ii = cp.int32(index) - cp.int32(jj * height)

    if (ii < width and jj < height):
        center_x = width // 2
        center_y = height // 2

        distance_from_center = cp.sqrt((center_x - ii)**2 + (center_y - jj)**2)

        if ((distance_from_center > min_freq) and (distance_from_center < max_freq )):
            output_fft_plane[jj, ii] = input_fft_plane[jj, ii]
        else:
            output_fft_plane[jj, ii] = 0.0 + 0.0j

@jit.rawkernel()
def calculate_rayleigh_sommerfeld_propagation_kernel(kernel: cp.ndarray, wavelength: float, magnification: float, pixel_size: float, width: int, height: int, propagation_distance: float):
    """
    Calculates the Rayleigh-Sommerfeld propagation kernel for wave propagation.

    Args:
        kernel (cp.ndarray): Output kernel (complex data) on GPU, representing the propagation kernel.
        wavelength (float): Wavelength of the light in the medium.
        magnification (float): Magnification factor for the pixel size.
        pixel_size (float): Physical size of a pixel in the input plane.
        width (int): Width of the kernel (in pixels).
        height (int): Height of the kernel (in pixels).
        propagation_distance (float): Distance for wave propagation.
    """
    
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    jj = cp.int32(index) // cp.int32(width)
    ii = cp.int32(index) - cp.int32(jj * height)

    if (ii < width and jj < height):

        wave_number = cp.float32(2.0 * cp.pi / wavelength)
        x = cp.float32( cp.int32(ii) - cp.int32(width // 2) )
        y = cp.float32( cp.int32(jj) - cp.int32(height // 2) )

        scaled_pixel_size= cp.float32(pixel_size / magnification)

        module_term = cp.float32(propagation_distance) / cp.float32(wavelength * (propagation_distance**2 + (x**2 + y**2) * scaled_pixel_size**2))
        phase_term = cp.float32(wave_number * cp.sqrt(propagation_distance**2 + (x**2 + y**2) * scaled_pixel_size**2))
        kernel[jj, ii] = cp.complex64(module_term * cp.exp(2.0j*cp.pi*phase_term))

    
@jit.rawkernel()
def calculate_angular_spectrum_propagation_kernel(kernel: cp.ndarray, wavelength: float, magnification: float, pixel_size: float, width: int, height: int, propagation_distance: float):
    """
    Calculates the angular spectrum propagation kernel for wave propagation.

    Args:
        kernel (cp.ndarray): Output kernel (complex data) on GPU, representing the propagation kernel.
        wavelength (float): Wavelength of the light in the medium.
        magnification (float): Magnification factor for the pixel size.
        pixel_size (float): Physical size of a pixel in the input plane.
        width (int): Width of the kernel (in pixels).
        height (int): Height of the kernel (in pixels).
        propagation_distance (float): Distance for wave propagation.
    """

    index = cp.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)

    jj = cp.int32(index // width)
    ii = cp.int32(index - jj * width)

    if (ii < width and jj < height):
        du = magnification / (pixel_size * cp.float32(width))
        dv = magnification / (pixel_size * cp.float32(height))

        offset_u = width // 2
        offset_v = height // 2

        u = ( cp.int32(ii) - cp.int32(offset_u))*du
        v = ( cp.int32(jj) - cp.int32(offset_v))*dv
        
        argument_term = 1.0 - cp.square(wavelength * u) - cp.square(wavelength *v)
        
        if (argument_term > 0):
            kernel[jj, ii] = cp.exp(2 * 1j * cp.pi * propagation_distance * cp.sqrt(argument_term) / wavelength)
        else:
            kernel[jj, ii] = 0.0+0j


@jit.rawkernel()
def propagate_fresnel_phase_based(input_wavefront: cp.ndarray, output_wavefront: cp.ndarray, wavelength: float, magnification: float, pixel_size: float, width: int, height: int, propagation_distance: float):
    """
    Propagates a wavefront using the Fresnel diffraction method (phase-based approach).

    Args:
        input_wavefront (cp.ndarray): Input wavefront (complex data) on GPU.
        output_wavefront (cp.ndarray): Output wavefront (complex data) on GPU, where the result is stored.
        wavelength (float): Wavelength of the wave in the medium (in meters).
        magnification (float): Magnification factor for the pixel size.
        pixel_size (float): Physical size of a pixel in the input plane (in meters).
        width (int): Width of the wavefront (in pixels).
        height (int): Height of the wavefront (in pixels).
        propagation_distance (float): Distance for wave propagation (in meters).
    """

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    jj = index // width
    ii = index - jj * width

    if (ii < width and jj < height):
        offset_x = width//2
        offset_y = height//2

        x = ii - offset_x
        y = jj - offset_y

        scaled_pixel_size_x = magnification * wavelength * propagation_distance / (width * pixel_size)
        scaled_pixel_size_y = magnification * wavelength * propagation_distance / (height * pixel_size)

        argument_term = (cp.pi * 1.0j / (wavelength * propagation_distance))  * (x**2 * scaled_pixel_size_x**2 + y**2 * scaled_pixel_size_y**2)
        module_term = 1.0j * cp.exp(2.0j*cp.pi*propagation_distance/wavelength)/(wavelength*propagation_distance)
        output_wavefront[jj, ii] = module_term * cp.exp(argument_term) * input_wavefront[jj, ii]

@jit.rawkernel()
def propagate_fresnel_phase_based_two(input_wavefront: cp.ndarray, output_wavefront: cp.ndarray, wavelength: float, magnification: float, pixel_size: float, width: int, height: int, propagation_distance: float):
    """
    Propagates a wavefront using the Fresnel diffraction method

    Args:
        input_wavefront (cp.ndarray): Input wavefront
        output_wavefront (cp.ndarray): Output wavefront, where the result is stored
        wavelength (float): Wavelength of the wave in the medium
        magnification (float): Magnification factor for the pixel size
        pixel_size (float): Physical size of a pixel in the input plane
        width (int): Width of the wavefront (in pixels).
        height (int): Height of the wavefront (in pixels).
        propagation_distance (float): Distance for wave propagation (in meters).
    """
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    jj = index // width
    ii = index - jj * width

    if (ii < width and jj < height):
        offset_x = width // 2
        offset_y = height // 2

        x = ii - offset_x
        y = jj - offset_y

        scaled_pixel_size_x = pixel_size / magnification
        scaled_pixel_size_y = pixel_size / magnification

        argument_term = (cp.pi * 1.0j / (wavelength * propagation_distance)) * (x**2 * scaled_pixel_size_x**2 + y**2 * scaled_pixel_size_y**2)
        output_wavefront[jj, ii] = cp.exp(argument_term) * input_wavefront[jj, ii]



#########################################################################################################################################
################################               propagations                  ############################################################
#########################################################################################################################################

def propagate_angular_spectrum(input_wavefront: cp.ndarray, kernel: cp.ndarray, wavelength: float,
                               magnification: float, pixel_size: float, width: int, height: int,
                               propagation_distance: float, min_frequency: float, max_frequency: float):
    """
    Propagates a wavefront using the angular spectrum method (2-FFT approach).

    Args:
        input_wavefront (cp.ndarray): Input wavefront (complex data) on GPU.
        kernel (cp.ndarray): Angular spectrum propagation kernel (complex data) on GPU.
        wavelength (float): Wavelength of the wave in the medium (in meters).
        magnification (float): Magnification factor for the pixel size.
        pixel_size (float): Physical size of a pixel in the input plane (in meters).
        width (int): Width of the wavefront (in pixels).
        height (int): Height of the wavefront (in pixels).
        propagation_distance (float): Distance for wave propagation (in meters).
        min_frequency_px (float): Minimum frequency for bandpass filtering (in pixels).
        max_frequency_px (float): Maximum frequency for bandpass filtering (in pixels).

    Returns:
        cp.ndarray: The propagated wavefront.

    Notes:
        - The angular spectrum method is suitable for short propagation distances.
        - The method uses two FFTs: one for the input wavefront and one for the inverse transform.
    """
    
    n_threads = 1024
    n_blocks = math.ceil(width * height // n_threads)
    fft_wavefront = cp.fft.fftshift(cp.fft.fft2(input_wavefront, norm='ortho'))

    # Apply a bandpass filter
    if ((min_frequency != 0) and (max_frequency != 0)):
        apply_bandpass_filter(fft_wavefront, fft_wavefront, width, height, min_frequency, max_frequency)

    calculate_angular_spectrum_propagation_kernel[n_blocks, n_threads](kernel, wavelength, magnification, pixel_size, width, height, propagation_distance)

    fft_propagated_wavefront = fft_wavefront * kernel
    # Compute the inverse FFT to obtain the propagated wavefront
    output_wavefront = cp.fft.ifft2(cp.fft.ifftshift(fft_propagated_wavefront), norm='ortho')
    return output_wavefront

# TODO: Documentation and typing
def propagate_fresnell(input_wavefront: cp.ndarray, input_wavefront_two: cp.ndarray, output_wavefront: cp.ndarray,
                               wavelength: float, magnification: float, pixel_size: float, width: int, height: int, propagation_distance: float):
    
    ### propagation de type Fresnell (méthode à 1 FFT): propagation à utiliser pour les distances grandes 
    ### dx et dy dépendent de la distance de propagation
    ### méthode pas encore testée...un peu merdique

    n_threads = 1024
    n_blocks = math.ceil (width * height // n_threads)
    propagate_fresnel_phase_based[n_blocks, n_threads](input_wavefront, input_wavefront_two, wavelength, magnification, pixel_size, width, height, propagation_distance)
    fft_wavefront = cp.fft.fftshift(cp.fft.fft2(input_wavefront_two))
    propagate_fresnel_phase_based_two[n_blocks, n_threads](fft_wavefront, output_wavefront, wavelength, magnification, pixel_size, width, height, propagation_distance)

# TODO: Documentation and typing
def propagate_rayleigh_sommerfeld(input_wavefront, kernel,
wavelength, magnification, pixel_size, width, height, propagation_distance):
    
    ### propagation de type Rayleigh Sommerfeld (méthode à 3 FFTs): propagation à utiliser pour les distances grandes 
    ### dx et dy ne dépendent pas de la distance de propagation

    n_threads = 1024
    n_blocks = math.ceil(width * height // n_threads)
    fft_wavefront = cp.fft.fftshift(cp.fft.fft2(input_wavefront, norm='ortho'))
    calculate_rayleigh_sommerfeld_propagation_kernel[n_blocks, n_threads](kernel, wavelength, magnification, pixel_size, width, height, propagation_distance)
    # fft_kernel = fftshift(fft2(kernel, norm = 'ortho'))
    fft_propagated_wavefront = fft_wavefront * kernel
    output_wavefront = cp.fft.fft2(cp.fft.fftshift(fft_propagated_wavefront), norm='ortho')
    return output_wavefront

#########################################################################################################################################
################################               Calculs volumes               ############################################################
#########################################################################################################################################

# TODO: Documentation and typing
def volume_propagate_angular_spectrum_complex(input_wavefront, kernel, volume_wavefront,
wavelength, magnification, pixel_size, width, height, initial_propagation_distance, propagation_delta, number_propagation, min_frequency, max_frequency):

    n_threads = 1024
    n_blocks = math.ceil(width * height // n_threads)

    fft_wavefront = cp.fft.fftshift(cp.fft.fft2(input_wavefront, norm='ortho'))

    if ((min_frequency != 0) and (max_frequency != 0)):
        apply_bandpass_filter[n_blocks, n_threads](fft_wavefront, fft_wavefront, width, height, min_frequency, max_frequency)

    for i in range(number_propagation):
        distance = initial_propagation_distance + i * propagation_delta
        calculate_angular_spectrum_propagation_kernel[n_blocks, n_threads](kernel, wavelength, magnification, pixel_size, width, height, distance)
        fft_propagated_wavefront = fft_wavefront * kernel
        volume_wavefront[:,:,i] = cp.fft.fft2(cp.fft.fftshift(fft_propagated_wavefront), norm='ortho')

def volume_propagate_angular_spectrum_to_module(input_wavefront, kernel, volume_module_wavefront,
wavelength, magnification, pixel_size, width, height, initial_propagation_distance, propagation_delta, number_propagation, min_frequency, max_frequency):
    n_threads = 1024
    n_blocks = math.ceil(width * height // n_threads)
    fft_wavefront = cp.fft.fftshift(cp.fft.fft2(input_wavefront, norm='ortho'))

    if ((min_frequency != 0) and (max_frequency != 0)):
        apply_bandpass_filter[n_blocks, n_threads](fft_wavefront, fft_wavefront, width, height, min_frequency, max_frequency)

    for i in range(number_propagation):
        distance = initial_propagation_distance + i *propagation_delta 
        calculate_angular_spectrum_propagation_kernel[n_blocks, n_threads](kernel, wavelength, magnification, pixel_size, width, height, distance)
        fft_propagated_wavefront = fft_wavefront * kernel
        output_wavefront = cp.fft.fft2(cp.fft.fftshift(fft_propagated_wavefront), norm='ortho')
        volume_module_wavefront[i,:,:] = cp.flip(cp.flip(cp.sqrt(cp.real(output_wavefront)**2 + cp.imag(output_wavefront)**2), axis=1), axis=0)


def volume_propagate_Rayleigh_Sommerfeld(input_wavefront, kernel, volume_module_wavefront,
wavelength, magnification, pixel_size, width, height, propagation_delta, number_propagation):

    n_threads = 1024
    n_blocks = math.ceil(width * height // n_threads)
    fft_wavefront = cp.fft.fftshift(cp.fft.fft2(input_wavefront, norm='ortho'))

    for i in range(number_propagation):
        distance = (i + 1) * propagation_delta
        calculate_rayleigh_sommerfeld_propagation_kernel[n_blocks, n_threads](kernel, wavelength, magnification, pixel_size, width, height, distance)
        fft_propagated_wavefront = fft_wavefront * kernel
        volume_module_wavefront[:,:,i] = cp.fft.fft2(cp.fft.fftshift(fft_propagated_wavefront), norm='ortho')

def volume_propagate_fresnell(input_wavefront, input_wavefront_temp, volume_wavefront,
wavelength, magnification, pixel_size, width, height, propagation_delta, number_propagation):
    n_threads = 1024
    n_blocks = math.ceil(width * height // n_threads)

    for i in range(number_propagation):
        distance = (i + 1) * propagation_delta
        propagate_fresnel_phase_based[n_blocks, n_threads](input_wavefront, input_wavefront_temp, wavelength, magnification, pixel_size, width, height, distance)
        fft_wavefront_temp = cp.fft.fftshift(cp.fft.fft2(input_wavefront_temp, norm='ortho'))
        propagate_fresnel_phase_based_two[n_blocks, n_threads](fft_wavefront_temp, volume_wavefront[:,:,i], wavelength, magnification, pixel_size, width, height, distance)

