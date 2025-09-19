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

# Custom layer MU: returns a trainable scalar parameter tiled to the spatial dimensions.
class MU(nn.Module):
    def __init__(self):
        super(MU, self).__init__()
        # Parameter shape (1,1,1) which will be broadcast to (channels, height, width)
        self.w = nn.Parameter(torch.randn(1, 1, 1))
        nn.init.xavier_normal_(self.w)

    def forward(self):
        # x assumed shape: (batch, channels, height, width)
        # return self.w.clamp(min=0).expand(1, x.size(1), x.size(2), x.size(3))
        return self.w.clamp(min=0)


# Custom soft-thresholding layer.
class SoftThreshold(nn.Module):
    def __init__(self):
        super(SoftThreshold, self).__init__()
        self.bias = nn.Parameter(torch.randn(1, 1, 1))
        nn.init.xavier_normal_(self.bias)

    def forward(self, x):
        bias_tile = self.bias.clamp(min=0).expand(1, x.size(1), x.size(2), x.size(3))
        return torch.sign(x) * F.relu(torch.abs(x) - bias_tile)


# Residual block similar to the Keras res_block.
class ResBlock(nn.Module):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + shortcut
        out = self.relu(out)
        return out