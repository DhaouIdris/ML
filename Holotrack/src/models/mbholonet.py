import torch
import torch.nn as nn

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

# Phase block implementing the proximal update.
class PhaseBlock(nn.Module):
    def __init__(self, filter_num):
        super(PhaseBlock, self).__init__()
        self.F = ResBlock(filter_num)
        self.soft_threshold = SoftThreshold()
        self.G = ResBlock(filter_num)

    def forward(self, v):
        o_forward = self.F(v)
        o_soft = self.soft_threshold(o_forward)
        o_next = self.G(o_soft)
        o_forward_backward = self.G(o_forward)
        stage_symloss = o_forward_backward - v
        return o_next, stage_symloss


# Main network converting MBHoloNet.
class MBHoloNet(nn.Module):
    def __init__(self, img_rows=128, img_cols=128, img_depths=5, phase_num=9,
                 loss_sym_param=2e-3):
        super(MBHoloNet, self).__init__()
        self.img_rows = img_rows # volume width
        self.img_cols = img_cols # volume height
        self.img_depths = img_depths # volume depth

        self.phase_num = phase_num

        self.loss_sym_param = loss_sym_param
        self.filter_num = img_depths

        # Create one MU layer and one phase block per iteration.
        self.mu_layers = nn.ModuleList([MU() for _ in range(phase_num)])
        self.phase_blocks = nn.ModuleList([PhaseBlock(self.filter_num) for _ in range(phase_num)])
        
        # Final normalization layers.
        self.batch_norm = nn.BatchNorm2d(self.img_depths)

    def backward_wave_prop(self, holo, otf3d):
        """
        Backward propagation. Propagate a 2D hologram to a 3D volume.
        """
        Nz = self.img_depths
        # If otf3d is a 3D array, add a batch dimension.
        if otf3d.dim() == 3:
            otf3d = otf3d.unsqueeze(0)  # Now shape is (1, depth, height, width)
        holo = holo.to(torch.complex64)
        otf3d = otf3d.to(torch.complex64)
        conj_otf3d = torch.conj(otf3d)

        # Perform iFT(FT(O)conj(OTF))
        holo_expand = holo.unsqueeze(1).expand(-1, Nz, -1, -1)
        # TODO: backward ???
        holo_expand_ft = torch.fft.fft2(holo_expand, norm="backward")
        field3d_ft = holo_expand_ft * conj_otf3d  # Broadcast multiplication.
        field3d = torch.fft.ifft2(field3d_ft, norm="backward")
        vol = field3d.real  # Enforce real constraint.
        return vol

    def forward(self, holo, otf3d, return_vols: bool=False):
        """
        holo: tensor of shape (batch, img_rows, img_cols) [float32]
        otf3d: tensor of shape (depth, img_rows, img_cols) if provided as a 3D array.
        """
        # TODO: add line for shape issues
        otf3d = otf3d.permute(0, 3, 2, 1)
        # Initial guess via backward propagation.
        v_current = self.backward_wave_prop(holo, otf3d)
        o_temp = self.backward_wave_prop(holo, otf3d)
        loss_constraint = 0.0

        if return_vols: volumes = [self.batch_norm(v_current).permute(0, 2, 3, 1)]
        for i in range(self.phase_num):
            # Equation 10 in paper

            # numerator:  b = F( alpha * At * I_h + v)
            mu = self.mu_layers[i]()
            o_add_v = mu * o_temp + v_current
            b = o_add_v.to(torch.complex64)
            numerator = FT2d(b)

            # denominator = FT(b) / (|OTF|^2 + 1)
            otf_square = (otf3d.abs() ** 2)
            otf_square = mu * otf_square
            denominator = (otf_square + 1.).to(torch.complex64)

            x_prime = numerator / denominator
            v_next = iFT2d(x_prime).real

            # Proximal: min_o 0.5 ||x -r||_2^2 + theta ||x||_1
            phase_block = self.phase_blocks[i]
            o_next, stage_symloss = phase_block(v_next)
            v_current = o_next
            loss_constraint += stage_symloss.pow(2).mean()
            if return_vols: volumes.append(self.batch_norm(v_current).permute(0, 2, 3, 1))

        loss_constraint = self.loss_sym_param * loss_constraint / self.phase_num
        v_current = self.batch_norm(v_current)

        v_current = v_current.permute(0, 2, 3, 1)
        if return_vols: return v_current, loss_constraint, volumes
        return v_current, loss_constraint


# Example usage:
if __name__ == '__main__':
    batch_size = 2
    img_rows, img_cols = 64, 64
    img_depths = 64

    # Dummy hologram: (batch, height, width)
    holo = torch.randn(batch_size, img_rows, img_cols)
    # Dummy otf3d as a 3D array: (depth, height, width)
    otf_real = torch.randn(img_depths, img_rows, img_cols)
    otf_imag = torch.randn(img_depths, img_rows, img_cols)
    otf3d = torch.complex(otf_real, otf_imag)
    print(otf3d.shape)

    model = MBHoloNet(img_rows, img_cols, img_depths)
    output, aux_loss = model(holo, otf3d)
    print("Output shape:", output.shape)
    print("Auxiliary loss:", aux_loss.item())
