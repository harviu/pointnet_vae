from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import random


class VAE(nn.Module):
    def __init__(self,vector_length,channel,pred_num=128):
        """
        input size: B * N * C
        """
        super(VAE, self).__init__()
        self.vector_length = vector_length
        self.num_channel = channel
        self.prediction_num = pred_num
        self.pointnet = nn.Sequential(
            nn.Linear(channel,64),
            nn.LeakyReLU(),

            nn.Linear(64, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 1024),
            nn.LeakyReLU(),
        )
        self.fc01 = nn.Linear(1024,1024)
        self.fc10 = nn.Linear(1024,self.vector_length)
        self.fc11 = nn.Linear(1024,self.vector_length)

        self.fc_decode = nn.Sequential(
            nn.Linear(self.vector_length,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024,self.prediction_num * channel),
        )

    def encode(self,x):
        batch_size = len(x)
        new_tensor = torch.zeros((0,1024))
        if x[0].is_cuda:
            new_tensor = new_tensor.cuda()
        for pc in x:
            pc = self.pointnet(pc)
            pc = torch.max(pc,dim=-2)[0]
            new_tensor = torch.cat((new_tensor,pc[None,:]),dim=0)

        x = new_tensor
        # n_points = x.shape[1]
        # n_channel = self.num_channel
        # x = x.view(-1,n_channel)
        # x = self.pointnet(x)
        # x = x.view(batch_size, n_points, -1)
        # x = torch.max(x,dim=-2)[0]
        x = self.fc01(x)
        x = F.leaky_relu(x)
        mu, logvar = self.fc10(x), self.fc11(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self,z):
        y = self.fc_decode(z)
        y = y.view(-1,self.prediction_num,self.num_channel)
        return y

    def loss(self,output, target,logvar,mu):
        # print(output.shape,target.shape)
        recon_loss = 0
        for i in range(len(output)):
            pc1 = output[i]
            pc2 = target[i]
            recon_loss += nn_distance(pc1,pc2)
        recon_loss /= len(output)
        recon_loss *= 1000

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print(recon_loss,KLD)
        return recon_loss+KLD

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar

def nn_distance(pc1,pc2):
    np1 = pc1.shape[0]
    np2 = pc2.shape[0]
    pc1 = pc1[:,None,:].repeat(1,np2,1)
    pc2 = pc2[None,:,:].repeat(np1,1,1)
    d = (pc1 - pc2)**2
    d = torch.sum(d,dim=-1)

    d1 = torch.min(d,dim=-1)[0]
    d2 = torch.min(d,dim=-2)[0]
    dis = torch.cat((d1,d2),dim=0)
    dis = torch.mean(dis)
    # print(dis)
    return dis

class pointnet_layer(nn.Module):
    def __init__(self):
        super(pointnet_layer, self).__init__()
        pass

    def forward(self):
        pass
