import random
import os
import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import numpy as np

from model import AE
from process_data import *

def inference(pd,model,batch_size,args):
    loader = DataLoader(pd, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        if args.have_label:
            latent_all = torch.zeros((len(pd),args.vector_length//2),dtype=torch.float32,device="cpu")
        else:
            latent_all = torch.zeros((len(pd),args.vector_length),dtype=torch.float32,device="cpu")
        predict_all = torch.zeros((len(pd),2),dtype=torch.float32,device="cpu")
        for i, d in enumerate(loader):
            # t1 = time.time()
            if isinstance(d,list):
                data = d[0][:,:,:args.dim].float().cuda()
                if args.have_label:
                    label = d[-1].cuda()
                if args.mode=="ball" :
                    mask = d[1].cuda()
            else:
                data = d[:,:,:args.dim].float().cuda()
            if args.mode=="knn":
                latent = model.encode(data) 
            elif args.mode=="ball":
                latent = model.encode(data,mask) 
            if args.have_label:
                latent = model.cls[:6](latent)
                predict = model.cls[6:](latent)
                predict_all[i*batch_size:(i+1)*batch_size] = predict.detach().cpu()
            latent_all[i*batch_size:(i+1)*batch_size] = latent.detach().cpu()
            # t2 = time.time()
            # print(t2-t1)
            if args.have_label:
                loss_fn = nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fn(predict,label)
                test_loss += loss.item()
            print("processed",i+1,"/",len(loader),end="\r")
        print()
        print("loss: ", test_loss/len(loader))
    if args.have_label:
        return latent_all,predict_all
    else:
        return latent_all

if __name__ == "__main__":
    # input parsing
    parser = argparse.ArgumentParser(description='PointNet')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--sample-size', type=int, default=1000, metavar='N',
                        help='sample size per file (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--have-label', action='store_true', default=False,
                        help='have label for classification instead of autoencoder')
    parser.add_argument('-p', '--phase', type=int,default=0,dest="phase",
                        help='phase')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-w', '--load', dest='load', type=str,
                        default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int,
                        default=16, help='vector length')
    parser.add_argument('-d', '--dim', dest='dim', type=int,
                        default=4, help='number of point dimensions')
    parser.add_argument('-b', '--ball', dest='ball', action='store_true',
                        default=False, help='train with ball surrounding')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning-rate')
    parser.add_argument('-s', dest='source', type=str, default="fpm", help='data source')
    parser.add_argument('-k', dest='k', type=int, default=256, help='k in knn')
    parser.add_argument('-r', dest='r', type=float, default=0.2, help='r in ball query')
    parser.add_argument("--result-dir", dest="result_dir", type=str, default="states", help='the directory to save the result')
    # parser.add_argument('--recon-length', dest='recon_length', type=int, default=256, help='r in ball query')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.mode = "ball" if args.ball else 'knn'
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = AE(args).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.85)

    if args.load:
        state_dict = torch.load(args.load)
        state = state_dict['state']
        config = state_dict['config']
        args = config
        model.load_state_dict(state)
        print('Model loaded from {}'.format(args.load))

    if args.phase == 0:
        # train
        print(args)
        if args.source == "fpm":
            file_list = collect_file(data_path+"/2016_scivis_fpm/0.44/run03/",args.source,shuffle=True)
        elif args.source == "cos":
            file_list = collect_file(data_path+"/ds14_scivis_0128/raw",args.source,shuffle=True)
        for epoch in range(1, args.epochs + 1):
            for i,f in enumerate(file_list):
                print("file in process: ",f)
                print("file processed {}/{}".format(i,len(file_list)))
                if args.source == "fpm":
                    data_source = vtk_reader(f)
                elif args.source == "cos":
                    data_source = sdf_reader(f)
                if args.have_label:
                    # read halo file if have label
                    dir_name, real_file_name = os.path.split(f)
                    timestep = int(real_file_name[-4:-2])
                    if timestep == 0: 
                        timestep = 100
                    halo_file_name = dir_name+"/../rockstar/out_{}.list".format(timestep-2)
                    halo_info = halo_reader(data_path+"/ds14_scivis_0128/rockstar/out_47.list")
                    pd = PointData(data_source,args,None,halo_info)
                else:
                    pd = PointData(data_source,args)
                loader = DataLoader(pd, batch_size=args.batch_size, shuffle=True, drop_last=True)
                model.train()
                train_loss = 0
                print("number of samples: ",len(pd))
                for i, d in enumerate(loader):
                    if isinstance(d,list):
                        data = d[0][:,:,:args.dim].float().to(device)
                        if args.have_label:
                            label = d[-1].to(device)
                        if args.mode=="ball" :
                            mask = d[1].to(device)
                    else:
                        data = d[:,:,:args.dim].float().to(device)
                    optimizer.zero_grad()
                    if args.mode=="knn":
                        recon_batch = model(data) 
                    elif args.mode=="ball":
                        recon_batch = model(data,mask) 
                    if args.have_label:
                        loss_fn = nn.CrossEntropyLoss(reduction="mean")
                        loss = loss_fn(recon_batch,label)
                    else:
                        loss = model.loss(recon_batch, data)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

                    if i % args.log_interval == 0:
                        print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                            epoch, 
                            100. * i / len(loader),
                            loss.item(),
                            ))
                    if i == len(loader)-1:
                        print('====> Epoch: {} Average loss: {:.4f}'.format(
                            epoch, train_loss / len(loader)))
                if not os.path.isdir(args.result_dir):
                    os.mkdir(args.result_dir)
                save_dict = {
                    "state": model.state_dict(),
                    "config":args,
                }
                torch.save(save_dict,args.result_dir+'/CP{}.pth'.format(epoch))
                print('Checkpoint {} saved !'.format(epoch))
            scheduler.step()

