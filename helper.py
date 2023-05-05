import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from models import Quantizer

def conv(in_channels, out_channels, kernel_size=5, stride=2, padding=True):
    if padding:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
    else:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )


def deconv(in_channels, out_channels, kernel_size=5, stride=2, padding=True):
    if padding:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )
    else:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1
        )

def gaussian_kernel1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()

def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])

def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x

def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)

class ScaleSpaceFlow_res(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, 4, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(4, out_planes, kernel_size=4, stride=1, padding=False),
                    nn.Tanh()
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, 4, kernel_size=4, stride=1, padding=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(4, mid_planes, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.res_encoder = Encoder(1, out_planes=dim)
        self.res_decoder = Decoder(1, in_planes=dim*2)

        self.motion_encoder = Encoder(2 * 1, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref):
        if not self.freeze_enc:
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_motion), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

            y_combine = torch.cat((y_res, y_motion), dim=1)
            x_res_hat = self.res_decoder(y_combine)
            x_rec = x_pred + x_res_hat

        return x_rec

    def forward_getmotion(self, x_cur, x_ref):
        # encode the motion information
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)

        # decode the space-scale flow information
        motion_info = self.motion_decoder(y_motion)
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)

        # y_combine
        y_combine = torch.cat((y_res, y_motion), dim=1)
        x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        return x_rec, motion_info, x_res_hat

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list


class ScaleSpaceFlow(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        single_bit=False
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.single_bit = single_bit

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.res_encoder = Encoder(2, out_planes=dim)
        self.P_encoder = Encoder_P(1, out_planes=192)
        self.res_decoder = Decoder(1, in_planes=dim + 192)

        self.P_encoder_MSE = Encoder_P(1, out_planes=192)
        self.res_decoder_MSE = Decoder(1, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * 1, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref):
        if not self.freeze_enc:
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if self.single_bit:
                y_res=0*y_res

            y_pred = self.P_encoder(x_pred)

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

            #OK this
            y_pred = self.P_encoder(x_pred)
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

        return y_res, x_pred

    def forward_dec_MSE(self, y_res, x_pred):
        with torch.no_grad():
            y_pred = self.P_encoder_MSE(torch.cat((x_pred, x_pred.detach()), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder_MSE(y_combine)
            x_rec = torch.sigmoid(x_res_hat)

        return x_rec

    def forward_dec(self, y_res, x_pred, x_hat):
        with torch.no_grad():
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)
            x_rec = torch.sigmoid(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(2, out_planes=192)

        self.res_encoder = Encoder(3, out_planes=dim)

        self.res_decoder = Decoder(1, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * 1, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)
            if x_hat == None:
                x_hat = x_pred #no conditioning, else we condition it.
            # residual
            x_res = torch.cat((x_cur, x_pred, x_hat), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)
                if x_hat == None:
                    x_hat = x_pred #no conditioning, else we condition it.

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

                #OK this

            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref, x_hat=None):
        with torch.no_grad():
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)
            y_motion = self.quantize_noise(y_motion)

            #Before this
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)
            if x_hat == None:
                x_hat = x_pred #no conditioning, else we condition it.

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            #OK this

        y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
        y_combine = torch.cat((y_res, y_pred), dim=1)

        return y_combine

    def forward_dec(self, y_combine):
        with torch.no_grad():
            x_res_hat = self.res_decoder(y_combine)
            x_rec = torch.sigmoid(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps_universal(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(2, out_planes=192)

        self.res_encoder = Encoder(2, out_planes=dim)

        self.res_decoder = Decoder(1, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * 1, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if x_hat == None:
                x_hat = x_pred.detach() #no conditioning, else we condition it.
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

                #OK this
                if x_hat == None:
                    x_hat = x_pred.detach() #no conditioning, else we condition it.
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)
            y_motion = self.quantize_noise(y_motion)

            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            y_pred = self.P_encoder(x_pred)
            y_combine = torch.cat((y_res, y_pred), dim=1)
        #x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        #x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return y_combine

    def forward_dec(self, y_combine):
        with torch.no_grad():
            x_res_hat = self.res_decoder(y_combine)
            x_rec = torch.sigmoid(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps_universal_3frames(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2,
        num_c=3,
        single_bit=False,
        activation=torch.tanh
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc
        self.activation= activation

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(3 * num_c, out_planes=192)

        self.res_encoder = Encoder(2 * num_c, out_planes=dim)

        self.res_decoder = Decoder(num_c, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * num_c, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat1=None, x_hat2=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if x_hat1 == None and x_hat2==None:
                x_hat1 = x_pred.detach() #no conditioning, else we condition it.
                x_hat2 = x_pred.detach()
            y_pred = self.P_encoder(torch.cat((x_hat1,x_hat2, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = self.activation(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

                #OK this
                if x_hat1 == None and x_hat2==None:
                    x_hat1 = x_pred.detach() #no conditioning, else we condition it.
                    x_hat2 = x_pred.detach()
            y_pred = self.P_encoder(torch.cat((x_hat1,x_hat2, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = self.activation(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)
            y_motion = self.quantize_noise(y_motion)

            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            y_pred = self.P_encoder(x_pred)
            y_combine = torch.cat((y_res, y_pred), dim=1)
        #x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        #x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return y_combine

    def forward_dec(self, y_combine):
        with torch.no_grad():
            x_res_hat = self.res_decoder(y_combine)
            x_rec = self.activation(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps_e2e_3frames(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2,
        num_c=3,
        single_bit=False,
        activation=torch.tanh
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc
        self.activation= activation

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(3 * num_c, out_planes=192)

        self.res_encoder = Encoder(2 * num_c, out_planes=dim)

        self.res_decoder = Decoder(num_c, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * num_c, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat1=None, x_hat2=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if x_hat1 == None and x_hat2==None:
                x_hat1 = x_pred.detach() #no conditioning, else we condition it.
                x_hat2 = x_pred.detach()
            y_pred = self.P_encoder(torch.cat((x_hat1,x_hat2, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = self.activation(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                #OK this
                if x_hat1 == None and x_hat2==None:
                    x_hat1 = x_pred.detach() #no conditioning, else we condition it.
                    x_hat2 = x_pred.detach()
                x_res = torch.cat((x_cur, x_pred, x_hat1, x_hat2), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)
                
            y_pred = self.P_encoder(torch.cat((x_hat1,x_hat2, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = self.activation(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)
            y_motion = self.quantize_noise(y_motion)

            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            y_pred = self.P_encoder(x_pred)
            y_combine = torch.cat((y_res, y_pred), dim=1)
        #x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        #x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return y_combine

    def forward_dec(self, y_combine):
        with torch.no_grad():
            x_res_hat = self.res_decoder(y_combine)
            x_rec = self.activation(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps_universal_KTH(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2,
        num_c=3
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(2 * num_c, out_planes=192)
        self.P_encoder_MSE = Encoder_P(2 * num_c, out_planes=192)

        self.res_encoder = Encoder(2 * num_c, out_planes=dim)

        self.res_decoder = Decoder(num_c, in_planes=dim + 192)
        self.res_decoder_MSE = Decoder(num_c, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * num_c, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if x_hat == None:
                x_hat = x_pred.detach() #no conditioning, else we condition it.
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.tanh(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

                #OK this
                if x_hat == None:
                    x_hat = x_pred.detach() #no conditioning, else we condition it.
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.tanh(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

        return y_res, x_pred

    def forward_dec_MSE(self, y_res, x_pred):
        with torch.no_grad():
            y_pred = self.P_encoder_MSE(torch.cat((x_pred, x_pred.detach()), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder_MSE(y_combine)
            x_rec = torch.tanh(x_res_hat)

        return x_rec

    def forward_dec(self, y_res, x_pred, x_hat):
        with torch.no_grad():
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)
            x_rec = torch.tanh(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps_universal_KTH_3frames(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2,
        num_c=3
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(3 * num_c, out_planes=192)

        self.res_encoder = Encoder(2 * num_c, out_planes=dim)

        self.res_decoder = Decoder(num_c, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * num_c, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat1=None, x_hat2=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if x_hat1 == None and x_hat2==None:
                x_hat1 = x_pred.detach() #no conditioning, else we condition it.
                x_hat2 = x_pred.detach()
            y_pred = self.P_encoder(torch.cat((x_hat1,x_hat2, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.tanh(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

                #OK this
                if x_hat1 == None and x_hat2==None:
                    x_hat1 = x_pred.detach() #no conditioning, else we condition it.
                    x_hat2 = x_pred.detach()
            y_pred = self.P_encoder(torch.cat((x_hat1,x_hat2, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.tanh(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)
            y_motion = self.quantize_noise(y_motion)

            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            y_pred = self.P_encoder(x_pred)
            y_combine = torch.cat((y_res, y_pred), dim=1)
        #x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        #x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return y_combine

    def forward_dec(self, y_combine):
        with torch.no_grad():
            x_res_hat = self.res_decoder(y_combine)
            x_rec = torch.sigmoid(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

class ScaleSpaceFlow_R1eps_universal_old(nn.Module):
    r"""Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(
        self,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
        dim=1,
        stochastic = False,
        quantize_latents = False,
        L=2, q_limits=(-1.0, 1.0),
        freeze_enc=False,
        T=2
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=4, stride=1, padding=False),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )

        class Encoder_P(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.BatchNorm2d(mid_planes),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv(mid_planes, out_planes, kernel_size=4, stride=1, padding=False)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents
        self.L=L
        self.q_limits=q_limits
        self.freeze_enc= freeze_enc

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)

        self.P_encoder = Encoder_P(2, out_planes=192)

        self.res_encoder = Encoder(2, out_planes=dim)

        self.res_decoder = Decoder(1, in_planes=dim + 192)

        self.motion_encoder = Encoder(2 * 1, out_planes=dim)
        self.motion_decoder = Decoder(2 + 1, in_planes=dim)

        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

        self.freeze_enc= freeze_enc

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x
    def quantize_noise(self, y):
        # Quantize
        noise = torch.zeros_like(y)

        if self.stochastic:
            noise += uniform_noise(y.size(), self.alpha).cuda()
            y = y + noise
        if self.quantize_latents:
            y = self.q(y)
        if self.stochastic:
            y = y - noise
        return y

    def forward(self, x_cur, x_ref, x_hat=None):
        if not self.freeze_enc:
            # x_ref is mse solution
            # encode the motion information
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)

            #print (y_motion.min())
            #print (y_motion.max())
            # Quantize
            y_motion = self.quantize_noise(y_motion)

            # decode the space-scale flow information
            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

            # residual
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            if x_hat == None:
                x_hat = x_pred #no conditioning, else we condition it.
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))

            #print ("Residual Info: ", y_res.shape)

            # y_combine
            y_combine = torch.cat((y_res, y_pred), dim=1)
            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat
        else:
            with torch.no_grad():
                x = torch.cat((x_cur, x_ref), dim=1)
                y_motion = self.motion_encoder(x)
                y_motion = self.quantize_noise(y_motion)

                #Before this
                motion_info = self.motion_decoder(y_motion)
                #print ("Motion Info: ", y_motion.shape)
                x_pred = self.forward_prediction(x_ref, motion_info)

            with torch.no_grad():
                x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
                y_res = self.res_encoder(x_res)
                y_res = self.quantize_noise(y_res)

                #OK this
                if x_hat == None:
                    x_hat = x_pred #no conditioning, else we condition it.
            y_pred = self.P_encoder(torch.cat((x_hat, x_pred), dim=1))
            y_combine = torch.cat((y_res, y_pred), dim=1)

            x_res_hat = self.res_decoder(y_combine)

            # final reconstruction: prediction + residual
            x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return x_rec

    def forward_enc(self, x_cur, x_ref):
        with torch.no_grad():
            x = torch.cat((x_cur, x_ref), dim=1)
            y_motion = self.motion_encoder(x)
            y_motion = self.quantize_noise(y_motion)

            motion_info = self.motion_decoder(y_motion)
            #print ("Motion Info: ", y_motion.shape)
            x_pred = self.forward_prediction(x_ref, motion_info)

        with torch.no_grad():
            x_res = torch.cat((x_cur, x_pred), dim=1)#x_cur - x_pred
            y_res = self.res_encoder(x_res)
            y_res = self.quantize_noise(y_res)

            y_pred = self.P_encoder(x_pred)
            y_combine = torch.cat((y_res, y_pred), dim=1)
        #x_res_hat = self.res_decoder(y_combine)

        # final reconstruction: prediction + residual
        #x_rec = torch.sigmoid(x_res_hat) #x_pred + x_res_hat

        return y_combine

    def forward_dec(self, y_combine):
        with torch.no_grad():
            x_res_hat = self.res_decoder(y_combine)
            x_rec = torch.sigmoid(x_res_hat)

        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32

        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(
                    interp, scale_factor=2, mode="bilinear", align_corners=False
                )
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    def warp_volume(self, volume, flow, scale_field, padding_mode: str = "border"):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(
                f"Invalid number of dimensions for volume {volume.ndimension()}"
            )

        N, C, _, H, W = volume.size()

        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)

        out = F.grid_sample(
            volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False
        )
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)

        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list
