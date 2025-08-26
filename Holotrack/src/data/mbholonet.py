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
