#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:16:35 2020

@author: irtaza

A simple implementation of the variational laplace autoencoder 
based on the local linear approximation of ReLu networks. 
Paper: Variational Laplace Autoencoders; Yookoon Park, Chris Kim, Gunhee Kim ; PMLR 97:5032-5041, 2019.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np


class Encoder(nn.Module):
    
    def __init__(self, zdim, indim, hdim):
        "how many layers do I need? All linear for the local pca approx to hold. Following prescription in the paper"
        
        super().__init__()
        
        self.hd1 = nn.Linear(indim, hdim)
        self.hd2 = nn.Linear(hdim, hdim)
        self.mu = nn.Linear(hdim, zdim)
        self.logvar = nn.Linear(hdim, zdim)

    def forward(self, x):
        x = F.relu(self.hd1(x))
        x = F.relu(self.hd2(x))
        
        mu, log_var = self.mu(x), self.logvar(x)
        
        return mu,log_var, x

# architecture exactly the same as above 
class Decoder(nn.Module):
    
    def __init__(self, zdim, outdim, hdim):
        
        super().__init__()
        self.hd1 = nn.Linear(zdim, hdim)
        self.hd2 = nn.Linear(hdim, hdim)
        self.mu = nn.Linear(hdim, outdim)
        self.logvar = nn.Parameter(torch.Tensor([0.0]))     
        
        
    def forward(self, x, final_forward=False):
        
        # mask computing step 
        
        if not final_forward:
            x = F.relu(self.hd1(x))
            mask = torch.unsqueeze(x>0, dim=-1).to(torch.float) # mask layer 1
            W = self.hd1.weight * mask
            
            
            x = F.relu(self.hd2(x))
            mask = torch.unsqueeze(x>0, dim=-1).to(torch.float) # mask layer 2
            W = torch.matmul(self.hd2.weight, W)
            W *= mask
        
            W = torch.matmul(self.mu.weight, W)
            
            mu = self.mu(x)
            
            return mu, self.logvar, W

        # non mask, normal feed forward
        
        else:
            x = F.relu(self.hd1(x))
            x = F.relu(self.hd2(x))
            
            mu, log_var = self.mu(x), self.logvar
            
            return mu,log_var
        
    

class VLAE(nn.Module):
    def __init__(self, image_size, zdim, hd_enc_dim, hd_dec_dim, update_alpha=0.5, updates=10):
    
        super().__init__()
        
        self.image_size = image_size
        self.prior = (torch.zeros(1,1), torch.zeros(1,1))    # gaussian(mean=0,logvariance=0)
        self.encoder = Encoder(zdim, self.image_size, hd_enc_dim)
        self.decoder = Decoder(zdim, self.image_size, hd_dec_dim)
        
        self.alpha = update_alpha
        self.updates = updates
        
        
    def mu_update(self, x, prev_mu, mu, logvar, decoder_weights):
        var_inv = torch.exp(-1*self.decoder.logvar).unsqueeze(1)
        covariance = torch.matmul(decoder_weights.transpose(1,2),decoder_weights)
        covariance *= var_inv
        covariance += torch.eye(covariance.shape[1]).unsqueeze(0)
        
        #print(mu.shape,torch.matmul(decoder_weights, prev_mu.unsqueeze(-1)).shape)
        bias = mu.unsqueeze(-1) - torch.matmul(decoder_weights, prev_mu.unsqueeze(-1))
        mu = torch.matmul(decoder_weights.transpose(1, 2) * var_inv, x.view(-1, self.image_size, 1) - bias)
        mu = torch.matmul(torch.inverse(covariance), mu)
        mu = mu.squeeze(-1)
    
        return mu, covariance
        
    def forward(self, x):
        prev_mu, _, _ = self.encoder.forward(x) #  initialize
        
        #print(prev_mu.shape)
        for i in range(self.updates):
            mu, logvar, decoder_weights = self.decoder.forward(prev_mu)
            
            new_mu, covariance = self.mu_update(x, prev_mu, mu, logvar, decoder_weights)
            
            prev_mu = (1 - self.alpha /(1+i))*prev_mu + (self.alpha /(1+i))*new_mu
            
        
        _, __, decoder_weights = self.decoder.forward(prev_mu)
        var_inv = torch.exp(-1*self.decoder.logvar).unsqueeze(1)
        covariance = torch.matmul(decoder_weights.transpose(1,2),decoder_weights)
        covariance *= var_inv
        covariance += torch.eye(covariance.shape[1]).unsqueeze(0)
        
        
        #q_z_x = distribution.Gaussian(mu, covariance)
        
        L = torch.cholesky(torch.inverse(covariance))
        epsilon = torch.rand_like(prev_mu).unsqueeze(-1)
        z = prev_mu + torch.matmul(L, epsilon).squeeze(-1) # reparametrization trick sampled on q_z_x
        
        q_z_x = (L, prev_mu, covariance)
        p_x_z = self.decoder.forward(z, final_forward=True)

        return self.elbo_loss(x, z, likelihood=p_x_z, prior=self.prior, posterior=q_z_x)
        

    
    def elbo_loss(self, x, z, likelihood, prior, posterior):
        p_z_mu, p_z_logvar = prior
        p_x_z_mu, p_x_z_logvar = likelihood
        
        p_x_z_logprobs = 0.5*torch.sum(np.log(2*np.pi) + p_x_z_logvar + (x.view(-1, self.image_size)-p_x_z_mu)**2/torch.exp(p_x_z_logvar), dim=1)
        p_z_logprobs = 0.5*torch.sum(np.log(2*np.pi) + p_z_logvar + (z-p_z_mu)**2/torch.exp(p_z_logvar), dim=1)
        
        
        L, q_z_x_mu, q_z_x_covar = posterior # target posterior; not true posterior
        dim = q_z_x_mu.shape[1]
        indices = np.arange(L.shape[-1])
        q_z_x_logprobs = -0.5 * (dim * np.log(2.0*np.pi)
                       + 2.0 * torch.log(L[:, indices, indices]).sum(1)
                       + torch.matmul(torch.matmul((z - q_z_x_mu).unsqueeze(1), q_z_x_covar),
                                      (z - q_z_x_mu).unsqueeze(-1)).sum([1, 2]))
        
        # elbo: xz + z - qzx; -1 for maximization opt
        return -1*torch.mean(p_x_z_logprobs + p_z_logprobs - q_z_x_logprobs)  
        
        


        
        
        