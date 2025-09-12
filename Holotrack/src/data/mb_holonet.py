import os
import json
import numpy as np

from src import utils

class MBHolonetParams:
    def __init__(self, **kwargs):
        self._save_params(**kwargs)
        self._get_params()
        dtype = getattr(self, "dtype", "float32")
        self.dtype = getattr(np, dtype)

    def _save_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_params(self):
        for n in ["z0", "pps", "dz", "wavelength"]:
            if hasattr(self, n): setattr(self, n, float(getattr(self, n)))
        if self.holo_data_type == 1:
            ppv_ranges = {3: (1e-3, 5e-3), 7: (1e-3, 5e-3), 15: (5e-4, 25e-4), 32: (2e-4, 1e-3), 30: (0.5e-3, 1e-3), 128: (0.5e-6, 1.e-6)}
            # ppv_ranges = {3: (1e-3, 5e-3), 7: (1e-3, 5e-3), 15: (5e-4, 25e-4), 32: (2e-4, 1e-3), 30: (0.1, 0.5)}
            self.ppv_min, self.ppv_max = ppv_ranges.get(self.nz, (1e-3, 5e-3))
        elif self.holo_data_type == 2:
            self.noise_levels = list(range(10, 55, 5))
            self.group_num = len(self.noise_levels)
            self.data_num = 100
            self.ppv_min, self.ppv_max = 1e-3, 5e-3
        elif self.holo_data_type == 3:
            self.ppvs = [i * 1e-3 for i in range(1, 11)]
            self.group_num = len(self.ppvs)
            self.data_num = 100
        elif self.holo_data_type == 4:
            self.nxy, self.nz, self.wavelength, self.pps, self.dz, self.z0 = 128, 128, 632e-9, 20e-6, 50e-6, 10e-3
            self.na = self.pps * self.nxy / 2 / self.z0
            self.delta_x, self.delta_z = self.wavelength / self.na, 2 * self.wavelength / (self.na ** 2)
            self.ppv_min, self.ppv_max = 1.9e-4 / 128, 6.1e-2 / 128
        else:
            self.na = self.pps*self.nxy/2/self.z0
            self.delta_x = self.wavelength/(self.na)
            self.delta_z = 2*self.wavelength/(self.na**2)

        self.z_range = self.z0 + np.arange(0, self.nz)*self.dz   # axial depth span of the object


class MBHolonetHologram:
    def __init__(self, params: MBHolonetParams):
        self.params = params

    @staticmethod
    def one_center_particle(nxy, nz, dtype):
        vol = np.zeros((nxy, nxy, nz), dtype=dtype)
        center_x = nxy // 2
        center_y = nxy // 2
        center_z = nz // 2 
        vol[center_x, center_y, center_z] = 1  # Place a single particle at the center
        return vol

    @staticmethod
    def random_scatter(nxy, nz, radius, num_particles, dtype):
        n_pad = 1
        nxy_pad = nxy - 2 * n_pad
        vol = np.zeros((nxy, nxy, nz), dtype=dtype)
        
        # Ensure at least one particle per slice if possible
        if num_particles >= nz:
            # Place one particle per slice
            x = np.random.randint(1, nxy_pad + 1, size=nz) + n_pad
            y = np.random.randint(1, nxy_pad + 1, size=nz) + n_pad
            z = np.arange(1, nz + 1)
            
            # Remaining particles
            if num_particles > nz:
                x_extra = np.random.randint(1, nxy_pad + 1, size=num_particles - nz) + n_pad
                y_extra = np.random.randint(1, nxy_pad + 1, size=num_particles - nz) + n_pad
                z_extra = np.random.randint(1, nz + 1, size=num_particles - nz)
                x = np.concatenate([x, x_extra])
                y = np.concatenate([y, y_extra])
                z = np.concatenate([z, z_extra])
        else:
            # Fewer particles than slices: scatter randomly
            x = np.random.randint(1, nxy_pad + 1, size=num_particles) + n_pad
            y = np.random.randint(1, nxy_pad + 1, size=num_particles) + n_pad
            z = np.random.randint(1, nz + 1, size=num_particles)
        
        # Filter invalid positions
        valid = (x >= 1) & (x <= nxy) & (y >= 1) & (y <= nxy) & (z >= 1) & (z <= nz)
        vol[x[valid] - 1, y[valid] - 1, z[valid] - 1] = 1
        
        return vol

    @staticmethod
    def forward_projection(volume, otf3d):
        fft_holo = np.fft.fft2(volume, axes=(0,1)) * otf3d
        fft_holo = np.sum(fft_holo, axis=2)
        return np.fft.ifft2(fft_holo)

    @staticmethod
    def kernel_projection(pps, wavelength, z_list, nx, ny):
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        fx = (np.mod(x + nx/2, nx) - np.floor(nx / 2)) / nx
        fy = (np.mod(y + ny/2, ny) - np.floor(ny / 2)) / ny
        term = (fx**2 + fy**2) * (wavelength / pps)**2
        final_term = - 1/2 * term - 1/8 * term**2 - 1/16 * term**3
        otf = [np.exp(1j * 2 * np.pi / wavelength * z * final_term) for z in z_list]
        otf = np.stack(otf, axis=2)
        return otf

    @staticmethod
    def additive_white_gaussian_noise(data, snr_db, noise_type='measured'):
        snr_linear = 10 ** (snr_db / 10.0)
        if noise_type == "measured":
            signal_power = np.mean(np.abs(data) ** 2)
        else:
            signal_power = 10**(1.0/10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=data.shape)
        return data + noise

    @staticmethod
    def awgn(signal, snr_db):
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(scale=np.sqrt(noise_power), size=signal.shape)
        return signal + noise

    @staticmethod
    def gabor_hologram(volume, otf3d, noise_level):
        holo_field = MBHolonetHologram.forward_projection(volume, otf3d)
        holo_no_noise = np.abs(holo_field) ** 2
        
        ref_field = MBHolonetHologram.forward_projection(np.ones((*volume.shape[:2], 1)), otf3d)
        ref_no_noise = np.abs(ref_field)**2
        
        # Normalize the hologram
        holo = (holo_no_noise - ref_no_noise) / (np.sqrt(ref_no_noise) + np.finfo(float).eps)

        # Add Gaussian noise
        holo = MBHolonetHologram.additive_white_gaussian_noise(holo, noise_level, noise_type='measured')
        return holo

    @staticmethod
    def backward_projection(hologram, otf):
        planes = np.zeros_like(otf, dtype=np.complex128)
        for i in range(planes.shape[2]):
            planes[:, :, i] = np.fft.ifft2(np.fft.fft2(hologram) * np.conj(otf[:, :, i]))
        # return np.abs(planes)
        return np.abs(planes)**2
        # return np.log(np.abs(planes))

    def make_holograms(self):
        name = "" if not hasattr(self.params, "name") else self.params.name
        save_dir = getattr(self.params, "save_dir", "data") 
        dtype=str(self.params.dtype).split(".")[-1].split("'")[0]
        os.makedirs(save_dir, exist_ok=True)
        data_dir = os.path.join(save_dir, f"mb-holonet-{name}-{dtype}")
        data_dir = utils.get_next_dir(data_dir)
        holograms_dir = os.path.join(data_dir, "holograms")
        labels_dir = os.path.join(data_dir, "positions")
        for name in [data_dir, holograms_dir, labels_dir]:
            os.makedirs(name, exist_ok=True)

        otf3d = self.kernel_projection(self.params.pps, self.params.wavelength,
                                       self.params.z_range, self.params.nxy, self.params.nxy)
        # data = np.zeros((self.params.data_num, self.params.nxy, self.params.nxy), dtype=self.params.dtype)
        # labels = np.zeros((self.params.data_num, self.params.nxy, self.params.nxy, self.params.nz), dtype=n.uint8)
        total=self.params.nxy**2 * self.params.nz
        for i in range(self.params.data_num):
            # TODO: control number of particles
            min_particles = round(self.params.ppv_min * self.params.nxy * self.params.nxy * self.params.nz)
            max_particles = round(self.params.ppv_max * self.params.nxy * self.params.nxy * self.params.nz)
            print(f"Range of partciles number: min={min_particles}, max={max_particles}")
            random_number = np.random.randint(min_particles, max_particles)
            print(f"[{i+1}/{self.params.data_num}] Proportion of particles: {random_number/(total)}, number of particles: {random_number}")
            if hasattr(self.params, "centered_particle") and self.params.centered_particle:
                volume=MBHolonetHologram.one_center_particle(self.params.nxy, self.params.nz, self.params.dtype)
            else:
                volume = self.random_scatter(self.params.nxy, self.params.nz, self.params.sr, random_number, self.params.dtype)
            volume_prime = 1-volume
            hologram = - self.gabor_hologram(volume_prime, otf3d, self.params.noise_level)
            # data[i, :, :] = hologram.astype(self.params.dtype)
            # labels[i, :, :] = volume.astype(np.uint8)
            with open(os.path.join(holograms_dir, f"h-{i}"+".npy"), "wb") as f:
                np.save(f, hologram.astype(self.params.dtype))
            with open(os.path.join(labels_dir, f"h-{i}"+".npy"), "wb") as f:
                np.save(f, volume.astype(np.uint8))

        # data = (data-np.mean(data)) / np.std(data)
        # for i in tqdm(range(self.params.data_num)):
        #     with open(os.path.join(holograms_dir, f"h-{i}"+".npy"), "wb") as f:
        #         np.save(f, data[i,:,:])
        #     with open(os.path.join(labels_dir, f"h-{i}"+".npy"), "wb") as f:
        #         np.save(f, labels[i,:,:])

        with open(os.path.join(data_dir, "otf3d"+".npy"), "wb") as f:
            np.save(f, otf3d)


        params = {k: str(getattr(self.params, k)) for k in dir(self.params) if not k.startswith("_")}
        for k in params:
            if isinstance(params[k], np.ndarray):
                params[k] = list(params[k])
        with open(os.path.join(data_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        print("Params:\n",params, "\n")
        print(f"Save dir is {data_dir}")

    @staticmethod
    def read(data_dir):
        output = {}
        output["data"] = []
        output["labels"] = []
        holograms_dir = os.path.join(data_dir, "holograms")
        labels_dir = os.path.join(data_dir, "positions")
        data_size = len(os.listdir(holograms_dir))
        for i in range(data_size):
            with open(os.path.join(holograms_dir, f"h-{i}"+".npy"), "rb") as f:
                output["data"].append(np.load(f))
            with open(os.path.join(labels_dir, f"h-{i}"+".npy"), "rb") as f:
                output["labels"].append(np.load(f))
        output["data"] = np.concatenate(output["data"])
        output["labels"] = np.concatenate(output["labels"])
        with open(os.path.join(data_dir, "otf3d"+".npy"), "rb") as f:
            output["otf3d"] = np.load(f)
        with open(os.path.join(data_dir, "params.json"), "r") as f:
            output["params"] = json.load(f)
        return output

