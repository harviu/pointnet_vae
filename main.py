import random
import os
import argparse
import pickle

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np

import vtk
from vtk import *
from vtk.util import numpy_support

from model.vae import VAE
from process_data import *
from latent_max import LatentMax
from mean_shift import *
from simple import show




def train(epoch):
    model.train()
    train_loss = 0
    for i, data in enumerate(loader.load_data(0,50000)):
        data = to_tensor_list(data,device,args.dim)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                epoch, 
                100. * i / (50000//args.batch_size),
                loss.item(),
                ))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / (i+1)))
    torch.save(model.state_dict(),'result/CP{}.pth'.format(epoch))
    print('Checkpoint {} saved !'.format(epoch))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader.load_data(50000)):
            data = to_tensor_list(data,device,args.dim)
            recon_batch = model(data)
            loss = loss_function(recon_batch, data)
            test_loss += loss.item()
            # scatter_3d(data[0].cpu())
            # scatter_3d(recon_batch[0].cpu())

    test_loss /= (i+1)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    

if __name__ == "__main__":
    # input parsing
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-p', '--phase', type=int,default=0,dest="phase",
                        help='phase')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-w', '--load', dest='load', type=str,
                        default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int,
                        default=1024, help='vector length')
    parser.add_argument('-d', '--dim', dest='dim', type=int,
                        default=4, help='number of point dimensions')
    # parser.add_argument('-b', '--ball', dest='ball', action='store_true',
    #                     default=False, help='train with ball surrounding')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning-rate')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        data_path = os.environ['data'] + "/2016_scivis_fpm/0.44/"
    except KeyError:
        data_path = './data/'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = VAE(args.vector_length,args.dim,256).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = model.loss

    # if args.ball:
    #     train_file = data_path+ "/ball_4_60000_normalized.npy"
    #     test_file = data_path+"/ball_4_10000_normalized.npy"
    # else:
    #     train_file = data_path+"/knn_128_4_60000.npy"
    #     test_file = data_path+"/knn_128_4_10000.npy"

    data_file = "./data/new_sample.npy"

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    if args.phase == 1:
        data_directory = os.path.join(data_path,"run41/030.vtu")
        data = data_reader(data_directory)
        data = data_to_numpy(data)
        data = data[:,:4]
        data[:,3] = (data[:,3]-2.39460057e+01)/55.08245731
        idx = torch.load("./run41_030/saved_idx")

        model.eval()
        with torch.no_grad():
            latent_all = torch.zeros((0,16))
            for i in range(0,len(data),args.batch_size):
                ix = idx[i:i+args.batch_size]
                tensor_list = []
                for x in ix:
                    numpy_datum = data[x]
                    numpy_datum -= np.mean(numpy_datum,0,keepdims=True)
                    tensor = torch.from_numpy(numpy_datum).float().cuda()
                    tensor_list.append(tensor)
                latent = model.encode(tensor_list)
                latent_all = torch.cat((latent_all,latent.detach().cpu()),0)
                print(latent_all.shape)
                # recon_batch = model(tensor_list)
                # pc1 = tensor_list[11].cpu()
                # pc2 = recon_batch[11].cpu()
                # pca = PCA(n_components=2)
                # pca.fit(np.concatenate((pc1,pc2),axis=0))
                # pc1_embedded = pca.transform(pc1)
                # pc2_embedded = pca.transform(pc2)
                # plt.scatter(pc1_embedded[:,0],pc1_embedded[:,1])
                # plt.show()
                # plt.scatter(pc2_embedded[:,0],pc2_embedded[:,1])
                # plt.show()
                # scatter_3d(pc1)
                # scatter_3d(pc2)
            torch.save(latent_all,"latent_all")
    elif args.phase == 0:
        loader = Loader(data_file,args.batch_size)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
