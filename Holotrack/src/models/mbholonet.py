import torch

# For FFT shift operations; PyTorch 1.8+ has these built in.
def fftshift2d(x):
    return torch.fft.fftshift(x, dim=(-2, -1))

def ifftshift2d(x):
    return torch.fft.ifftshift(x, dim=(-2, -1))

def FT2d(a_tensor):
    # Equivalent to: ifftshift(fft2(fftshift(a_tensor)))
    x_shifted = fftshift2d(a_tensor)
    x_fft = torch.fft.fft2(x_shifted, norm="backward")
    x_out = ifftshift2d(x_fft)
    return x_out

def iFT2d(a_tensor):
    # Equivalent to: ifftshift(ifft2(fftshift(a_tensor)))
    x_shifted = fftshift2d(a_tensor)
    x_ifft = torch.fft.ifft2(x_shifted, norm="backward")
    x_out = ifftshift2d(x_ifft)
    return x_out