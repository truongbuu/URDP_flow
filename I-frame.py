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
from models import *
from utils import *
import torch.nn.functional as F

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

import argparse

parser = argparse.ArgumentParser('training config')
parser.add_argument('--total_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--lambda_gp', type=int, default=10, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='common_dim')
parser.add_argument('--z_dim', type=int, default=4, help='z dim')
parser.add_argument('--L', type=int, default=16, help='L size')
parser.add_argument('--skip_fq', type=int, default=5, help='loop frequency for WGAN')
parser.add_argument('--d_penalty', type=float, default=0.0, help='diversity penalty')
parser.add_argument('--lambda_P', type=float, default=0.0, help='Perceptual Penalty, keep at 1.0')
parser.add_argument('--lambda_PM', type=float, default=1.0, help='Perceptual Penalty Marginal, keep at 1.0')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Perceptual Penalty')


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0),  1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake[:,0],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def cal_W1(encoder, decoder, discriminator_M, test_loader):
    mse_loss = nn.MSELoss(reduction='sum')

    encoder.eval()
    decoder.eval()
    discriminator_M.eval()
    
    W1M_distance = []
    MSE = []

    num_x = 0
    for i, x in enumerate(iter(test_loader)):
        with torch.no_grad():
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            hx = encoder(x[:,:,0,:,:])
            x_hat =  decoder(hx[0])
            
            fake_img = x_hat.detach()
            real_img = x[:,0,:1,...].detach()
            fake_valid_m = discriminator_M(fake_img)
            real_valid_m = discriminator_M(real_img)

            W1M_distance.append(torch.sum(real_valid_m) - torch.sum(fake_valid_m))
            #print (F.mse_loss(x[:,:,1,:,:], x_hat)* x.size()[0])
            MSE.append(mse_loss(x[:,:,0,:,:], x_hat))
            #print (mse_loss(x[:,:,1,:,:], x_hat)/(64*64*len(x)))
            num_x += len(x)

    W1M_distance = torch.Tensor(W1M_distance)
    MSE = torch.Tensor(MSE)

    return W1M_distance.sum()/num_x, MSE.sum()/(64*64*num_x)

def main():
    args = parser.parse_args()
    #Params
    dim = args.dim#128
    z_dim = args.z_dim #2
    lambda_gp = args.lambda_gp #50
    bs = args.bs #64
    d_penalty = args.d_penalty #0
    skip_fq = args.skip_fq #10
    total_epochs = args.total_epochs #200
    lambda_P = args.lambda_P
    lambda_PM = args.lambda_PM
    lambda_MSE = args.lambda_MSE
    L = args.L
    
    #Create folder:
    folder_name='I_FRAME_dim_'+str(dim)+'|z_dim_'+str(z_dim)+'|L_'+str(L) + '|lambda_gp_'+str(lambda_gp) \
        +'|bs_'+str(bs)+'|dpenalty_'+str(d_penalty)+'|lambdaP_'+str(lambda_P)+'|lambdaPM_'+str(lambda_PM)+'|lambdaMSE_' + str(lambda_MSE)
    print ("Settings: ", folder_name)

    os.makedirs('./saved_models/'+ folder_name, exist_ok=True)
    f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
    
    encoder = Encoder(dim=z_dim, nc=1, stochastic=True, quantize_latents=True, L=L) #Generator Side
    decoder = Decoder_Iframe(dim=z_dim) #Generator Side
    discriminator_M = Discriminator_v3(out_ch=1) #Marginal Discriminator
    encoder.cuda()
    decoder.cuda()
    discriminator_M.cuda()
    train_loader, test_loader = get_dataloader(data_root='./data/', seq_len=2, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()
    
    opt_e = torch.optim.RMSprop(encoder.parameters(), lr=1e-4)
    opt_g = torch.optim.RMSprop(decoder.parameters(), lr=1e-4)
    opt_dm = torch.optim.RMSprop(discriminator_M.parameters(), lr=1e-4) #Marginal Discriminator
    
    
    for epoch in range(total_epochs):
        encoder.train()
        decoder.train()
        discriminator_M.train()
        for i,x in enumerate(iter(train_loader)):
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            hx = encoder(x[:,:,0,:,:])
            x_hat =  decoder(hx[0])

            opt_dm.zero_grad()
            fake_img = x_hat.detach()
            real_img = x[:,0,:1,...].detach()
            fake_valid_m = discriminator_M(fake_img)
            real_valid_m = discriminator_M(real_img)
            gradient_penalty_m = compute_gradient_penalty(discriminator_M, fake_img.data, real_img.data)
            errID =  -torch.mean(real_valid_m) + torch.mean(fake_valid_m) + lambda_gp * gradient_penalty_m
            errID.backward()
            opt_dm.step()

            if i%skip_fq == 0:
                for j in range(1):
                    #Generator
                    inp = x[:,:,0,:,:].detach()
                    hx = encoder(inp)
                    x_hat =  decoder(hx[0])

                    fake_vid = torch.cat((inp, x_hat), dim = 1)
                    fake_img = x_hat

                    opt_e.zero_grad()
                    opt_g.zero_grad()

                    #IG_fake = discriminator(fake_vid)
                    fake_validity_im = discriminator_M(fake_img)
                    errIG = -torch.mean(fake_validity_im)
                    #print (errIG)
                    loss =  lambda_PM*errIG +lambda_MSE*mse(x_hat, x[:,:,0,:,:])
                    #print (mse(x_hat, x[:,:,1,:,:]))
                    loss.backward()
                    opt_e.step()
                    opt_g.step()
                    #print (loss)
        if epoch % 20 ==0:
            show_str= "Epoch: "+ str(epoch) + "l_PM, l_MSE" + str(lambda_PM)+ " " +str(lambda_MSE) + " P loss: " + str(cal_W1(encoder, decoder, discriminator_M, test_loader)) + "MSE: " + str(mse(x_hat, x[:,:,0,:,:]))
            print(show_str) 
            f.write(show_str+"\n")
    
    show_str= "Epoch: "+ str(epoch) + "l_PM, l_MSE" + str(lambda_PM)+ " " +str(lambda_MSE) + " P loss: " + str(cal_W1(encoder, decoder, discriminator_M, test_loader)) + "MSE: " + str(mse(x_hat, x[:,:,0,:,:]))
    print(show_str) 
    f.write(show_str+"\n")
            
    encoder.eval()
    decoder.eval()
    discriminator_M.eval()
    
    torch.save(encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'decoder.pth'))
    torch.save(discriminator_M.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_M.pth'))
    
    
    
if __name__ == "__main__":
    main()
