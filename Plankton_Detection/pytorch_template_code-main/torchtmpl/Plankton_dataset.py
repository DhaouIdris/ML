import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def show_plankton_image(img, mask):
    """
    Display an image and its mask side by side

    img is either (H, W, 1)
    mask is either (H, W)
    """

    img = img.squeeze()

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, interpolation="none", cmap="tab20c")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("plankton_sample.png", bbox_inches="tight", dpi=300)
    plt.show()

def extract_patch_from_ppm(ppm_path, row_idx, col_idx, patch_size):
    """
    Extract a patch from a PPM image

    Arguments:
    - ppm_path: the path to the PPM image
    - row_idx: the row index of the patch
    - col_idx: the column index of the patch
    - patch_size: the size of the patch

    Returns:
    - patch: the extracted patch
    """
    # Read the PPM image and extract the patch
    with open(ppm_path, "rb") as f:
        # Skip the PPM magic number
        f.readline()
        # Skip the PPM comment
        while True:
            line = f.readline().decode("utf-8")
            if not line.startswith("#"):
                break
        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))
        
        # Maxval is either lower than 256 or 65536
        # It is actually 255 for the scans, and 65536 for the masks
        # This maximal value impacts the number of bytes used for encoding the pixels' value
        if maxval == 255:
            nbytes_per_pixel = 1
            dtype = np.uint8
        elif maxval == 65535:
            nbytes_per_pixel = 2
            dtype = np.dtype("uint16")
            # The PPM image is in big endian
            dtype = dtype.newbyteorder(">")
        else:
            raise ValueError(f"Unsupported maxval {maxval}")

        first_pixel_offset = f.tell()
        f.seek(0, 2)  # Seek to the end of the file
        data_size = f.tell() - first_pixel_offset
        # Check that the file size is as expected
        assert data_size == (ncols * nrows * nbytes_per_pixel)

        f.seek(first_pixel_offset)  # Seek back to the first pixel

         # Adjust the patch size if it goes beyond the image boundaries
        row_end = row_idx + patch_size[0]
        col_end = col_idx + patch_size[1]
        if row_end > nrows:
            patch_size = (nrows - row_idx, patch_size[1])
            row_end = nrows
        if col_end > ncols:
            patch_size = (patch_size[0], ncols - col_idx)
            col_end = ncols
        
        # Read all the rows of the patch from the image
        patch = np.zeros(patch_size, dtype=dtype)
        for i in range(patch_size[0]):
            f.seek(
                first_pixel_offset
                + ((row_idx + i) * ncols + col_idx) * nbytes_per_pixel,
                0,  # whence
            )
            row_data = f.read(patch_size[1] * nbytes_per_pixel)
            patch[i] = np.frombuffer(row_data, dtype=dtype)

    return patch

def get_size(ppm_path):
    with open(ppm_path, "rb") as f:
        # Skip the PPM magic number
        f.readline()
        # Skip the PPM comment
        while True:
            line = f.readline().decode("utf-8")
            if not line.startswith("#"):
                break
        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))
        
        # Maxval is either lower than 256 or 65536
        # It is actually 255 for the scans, and 65536 for the masks
        # This maximal value impacts the number of bytes used for encoding the pixels' value
        if maxval == 255:
            nbytes_per_pixel = 1
            dtype = np.uint8
        elif maxval == 65535:
            nbytes_per_pixel = 2
            dtype = np.dtype("uint16").newbyteorder(">")
        else:
            raise ValueError(f"Unsupported maxval {maxval}")

        first_pixel_offset = f.tell()
        f.seek(0, 2)   # Seek to the end of the file
        data_size = f.tell() - first_pixel_offset
        assert data_size == (ncols * nrows * nbytes_per_pixel)
    
    return nrows, ncols


class PlanktonDataset(Dataset):
    def __init__(self, dir, patch_size, stride, train=True, transform = None):
        self.dir = dir
        self.patch_size = patch_size
        self.stride = stride
        self.train = train
        self.transform = transform
        self.scan_files = []
        self.mask_files = []
        self.patches = []
        self.image_sizes = {} 
        
        for file_name in os.listdir(dir):
            if file_name.endswith("scan.png.ppm"):
                base_name = file_name.replace("scan.png.ppm", "")
                mask_name = base_name + "mask.png.ppm"

                scan_path = os.path.join(dir, file_name)
                mask_path = os.path.join(dir, mask_name)

                if os.path.exists(scan_path):
                    self.scan_files.append(scan_path)
                if os.path.exists(mask_path):
                    self.mask_files.append(mask_path)

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        img_idx, patch_i, patch_j = self.patches[idx]
        
        row_start = max(0, min(patch_i, self.image_sizes[img_idx][1] - self.patch_size))
        col_start = max(0, min(patch_j, self.image_sizes[img_idx][0] - self.patch_size))
        img_patch = extract_patch_from_ppm(self.scan_files[img_idx], row_start, col_start, (self.patch_size, self.patch_size))
        
        if self.train:
            mask_patch = extract_patch_from_ppm(self.mask_files[img_idx], row_start, col_start, (self.patch_size, self.patch_size))
            mask_patch = np.where(mask_patch < 8, 0, 1).astype(np.float32)
            if mask_patch.dtype.byteorder not in ('=', '|'):
                mask_patch = mask_patch.astype(mask_patch.dtype.newbyteorder('='))
