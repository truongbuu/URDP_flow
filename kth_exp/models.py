import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
from utils import *
import matplotlib.pyplot as plt
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init
import torchvision
import torch
import torch.nn as nn
from utils import uniform_noise, generate_centers

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Quantizer(nn.Module):
    """
    Scalar Quantizer module
    Source: https://github.com/mitscha/dplc
    """

    def __init__(self, centers=[-1.0, 1.0], sigma=1.0):
        super(Quantizer, self).__init__()
        self.centers = centers
        self.sigma = sigma

    def forward(self, x):
        centers = x.data.new(self.centers)
        xsize = list(x.size())

        # Compute differentiable soft quantized version
        x = x.view(*(xsize + [1]))
        level_var = Variable(centers, requires_grad=False)
        # dist = torch.pow(x-level_var, 2)
        dist = torch.abs(x-level_var)
        output = torch.sum(
            level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)
        # print(centers)
        # print(dist)
        # print(output)

        # Compute hard quantization (invisible to autograd)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)):
            centers.unsqueeze(0)  # in-place error
        centers = centers.expand(*(xsize + [len(self.centers)]))

        quant = centers.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv_linear(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv_linear, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    def __init__(self, dim, nc=1, stochastic=False,
                 quantize_latents=False, L=2, q_limits=(-1.0, 1.0),):
        super(Encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents

        #Set alpha
        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)


    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x

    def add_stochasticity(self, x, u):
        assert self.stochastic, f'Stochasticity disabled'

        return x + u
    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)

        noise = torch.zeros_like(h5)
        #uniform_noise(h5.size(), self.alpha).cuda()
        if self.stochastic:
            noise += uniform_noise(h5.size(), self.alpha).cuda()
            h5 = h5 +noise
        if self.quantize_latents:
            h5 = self.q(h5)
        if self.stochastic:
            h5 = h5 -noise

        return h5.view(-1, self.dim), [h1, h2, h3, h4]
    
class Encoder_128(nn.Module):
    def __init__(self, dim, nc=1, stochastic=False,
                 quantize_latents=False, L=2, q_limits=(-1.0, 1.0),):
        super(Encoder_128, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, 256)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(256, 256)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(256, 256)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(256, 256)
        self.c5 = dcgan_conv(256, 256)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(256, dim, 4, 1, 0),
                nn.BatchNorm2d(dim)
                )

        self.stochastic = stochastic
        self.quantize_latents = quantize_latents

        #Set alpha
        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)

        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)


    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x

    def add_stochasticity(self, x, u):
        assert self.stochastic, f'Stochasticity disabled'

        return x + u
    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h5 = self.c6(h5)
        
        """
        noise = torch.zeros_like(h5)
        #uniform_noise(h5.size(), self.alpha).cuda()
        if self.stochastic:
            noise += uniform_noise(h5.size(), self.alpha).cuda()
            h5 = h5 +noise
        if self.quantize_latents:
            h5 = self.q(h5)
        if self.stochastic:
            h5 = h5 -noise
        """

        return h5.view(-1, self.dim), [h1, h2, h3, h4]

class Decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(Decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output

class Decoder_Iframe(nn.Module):
    def __init__(self, dim, nc=1):
        super(Decoder_Iframe, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 , nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 , nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 , nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf , nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        vec = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1], 1))
        d3 = self.upc3(torch.cat([d2], 1))
        d4 = self.upc4(torch.cat([d3], 1))
        output = self.upc5(torch.cat([d4], 1))
        return output

class Decoder_noisy(nn.Module):
    def __init__(self, dim, nc=1):
        super(Decoder_noisy, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                #nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc1_ = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                #nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv_linear(nf * 8 , nf * 4)
        self.upc2_ = dcgan_upconv_linear(nf * 8 , nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv_linear(nf * 4 , nf * 2)
        self.upc3_ = dcgan_upconv_linear(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv_linear(nf * 2 , nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf , nc, 4, 2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        vec, skip = input

        d1  = self.upc1(vec.view(-1, self.dim, 1, 1))
        noise_1 = torch.randn_like(vec.view(-1, self.dim, 1, 1))
        d1_ = self.upc1_(vec.view(-1, self.dim, 1, 1)+ noise_1)
        d1 = d1_ + (d1 * d1_)

        d2 = self.upc2(torch.cat([d1], 1))
        noise_2 = torch.randn_like(d1)
        d2_ = self.upc2_(torch.cat([d1], 1) + noise_2)
        d2_ = d2_ + (d2 * d2_)

        d3 = self.upc3(torch.cat([d2], 1))
        noise_3 = torch.randn_like(d2)
        d3_ = self.upc3_(torch.cat([d2], 1) + noise_3)
        d3 = d3_ + (d3*d3_)

        d4 = self.upc4(torch.cat([d3], 1))
        output = self.upc5(torch.cat([d4], 1))
        return output

class Decoder_noisy2(nn.Module):
    def __init__(self, dim, nc=1):
        super(Decoder_noisy2, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                #nn.LeakyReLU(0.2, inplace=True)
                )

        self.upc1_ = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                #nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv_linear(nf * 8 * 2 , nf * 4)
        self.upc2_ = dcgan_upconv_linear(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv_linear(nf * 4 * 2, nf * 2)
        self.upc3_ = dcgan_upconv_linear(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv_linear(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2 , nc, 4, 2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        vec, skip, z = input

        d1  = self.upc1(vec.view(-1, self.dim, 1, 1))
        noise_1 = torch.randn_like(vec.view(-1, self.dim, 1, 1))
        d1_ = self.upc1_(vec.view(-1, self.dim, 1, 1)+ noise_1)
        d1 = d1_ + (d1 * d1_)

        d2 = self.upc2(torch.cat([d1], 1))
        noise_2 = torch.randn_like(d1)
        d2_ = self.upc2_(torch.cat([d1, skip[3]], 1) + noise_2)
        d2_ = d2_ + (d2 * d2_)

        d3 = self.upc3(torch.cat([d2], 1))
        noise_3 = torch.randn_like(d2)
        d3_ = self.upc3_(torch.cat([d2, skip[2]], 1) + noise_3)
        d3 = d3_ + (d3*d3_)

        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output

class Fuser(nn.Module):
    def __init__(self, dim, z_dim):
        super(Fuser, self).__init__()
        self.ln1 = nn.Linear(dim + z_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 128)
        self.activation = nn.Tanh()

    def forward(self, input):
        x = self.ln1(input)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.activation(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ch=64, out_ch=2):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            spectralnorm(nn.Conv2d(out_ch, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch, ch*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch*2, ch*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch*4, ch*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch*8, 1, 4, 1, 0)),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.normal_(module.weight, 0, 0.02)

    def forward(self, x):

        out = self.net(x)

        return out.squeeze(-1).squeeze(-1).squeeze(-1)

class Discriminator_v3(nn.Module):
    def __init__(self, ch=64, out_ch=2):
        super(Discriminator_v3, self).__init__()

        self.net = nn.Sequential(
            spectralnorm(nn.Conv2d(out_ch, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch, ch*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch*2, ch*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch*4, ch*8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch*8, 1, 4, 1, 0),
        )

        """self.net = nn.Sequential(
            nn.Conv2d(out_ch, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch, ch*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch*2, ch*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectralnorm(nn.Conv2d(ch*4, ch*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch*8, 1, 4, 1, 0),
        )"""

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.normal_(module.weight, 0, 0.02)

    def forward(self, x):

        out = self.net(x)

        return out.squeeze(-1).squeeze(-1).squeeze(-1)

class Discriminator_Iframe(nn.Module):
    def __init__(self, ch=64, out_ch=1):
        super(Discriminator_Iframe, self).__init__()
        im_w = 64
        im_h = 64
        self.net = nn.Sequential(
            nn.Conv2d(out_ch, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([ch, im_w//2, im_h//2]),

            nn.Conv2d(ch, ch*2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([ch*2, im_w//4, im_h//4]),

            nn.Conv2d(ch*2, ch*4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([ch*4, im_w//8, im_h//8]),

            nn.Conv2d(ch*4, ch*8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([ch*8, im_w//16, im_h//16]),

            nn.Conv2d(ch*8, 1, 4, 1, 0),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.normal_(module.weight, 0, 0.02)

    def forward(self, x):

        out = self.net(x)

        return out.squeeze(-1).squeeze(-1).squeeze(-1)

class Discriminator_KTH_dataset_wgan(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator_KTH_dataset_wgan, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            spectralnorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectralnorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectralnorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectralnorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
class Discriminator_KTH_dcgan(nn.Module):
    def __init__(self, ngpu=1,nc=3, ndf=64):
        super(Discriminator_KTH_dcgan, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class Discriminator_KTH_dataset_wgan_no_spectral(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator_KTH_dataset_wgan_no_spectral, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)