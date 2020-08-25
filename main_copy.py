import random
import os
import argparse
import pickle
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np

from model.vae import VAE
from model.dgcnn import SSAE
from process_data import FPM,wrap_data,scatter_3d
from sklearn.cluster import KMeans



def train():
    epoch = 0
    model.train()
    for run in os.listdir(data_path):
        for time_step in os.listdir(os.path.join(data_path,run)):
            if time_step.endswith(".vtu") and time_step == "020.vtu":
                epoch += 1 
                data_directory = os.path.join(data_path,run,time_step)
                fpm = FPM(data_directory)
                loader = DataLoader(fpm, batch_size=args.batch_size, shuffle=True, drop_last=True)

                train_loss = 0
                for i,data in enumerate(loader):
                    data = wrap_data(data.cuda())
                    optimizer.zero_grad()
                    z,y_cls,y_rec = model(data)
                    rec_loss, clu_loss, cls_loss = model.loss(data[0],z,y_cls,y_rec)
                    loss = rec_loss+clu_loss + cls_loss
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()

                    if i % args.log_interval == 0:
                        print('Train Epoch: {} [{:.0f}%]\tRec_Loss: {:.6f}\tClu_Loss: {:.6f}\tCls_Loss: {:.6f}'.format(
                            epoch, 
                            100. * i / (len(fpm)//args.batch_size),
                            rec_loss.item(),clu_loss.item(),cls_loss.item(),
                            ))

                model.update_center()

                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / (i+1)))
                save_dict = {
                    "state": model.state_dict(),
                    "center": model.centers,
                }
                torch.save(save_dict,'result/CP{}.pth'.format(epoch))
                print('Checkpoint {} saved !'.format(epoch))


def test(di1):
    fpm = FPM(di1)

    loader = DataLoader(fpm, batch_size=args.batch_size, shuffle=True, drop_last=True)
    epoch = 0
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for i,data in enumerate(loader):
            data = wrap_data(data.cuda())
            z,y_cls,y_rec = model(data)
            rec_loss, clu_loss, cls_loss = model.loss(data[0],z,y_cls,y_rec)
            loss = rec_loss + clu_loss + cls_loss
            test_loss += loss.item()

            if i % args.log_interval == 0:
                print('Train Epoch: {} [{:.0f}%]\tRec_Loss: {:.6f}\tClu_Loss: {:.6f}\tCls_Loss: {:.6f}'.format(
                    epoch, 
                    100. * i / (len(fpm)//args.batch_size),
                    rec_loss.item(),clu_loss.item(),cls_loss.item(),
                    ))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, test_loss / (i+1)))
    test_loss /= (i+1)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    print(model.z_acc.shape)
    fitter = KMeans(n_clusters=10)
    res = fitter.fit_predict(model.z_acc.cpu())
    # fitter.cluster_centers_ = model.centers.cpu()
    # res = fitter.predict(model.z_acc.cpu())
    torch.save(res,"saved_idx")
    res = torch.load("saved_idx")
    for i in range(args.cluster):
        print(np.sum(res==i))
        coord = fpm[:len(res),:3,0]
        show = np.concatenate((coord,res[:,None]),-1)
        scatter_3d(show[res==i])
    

if __name__ == "__main__":
    # input parsing
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=43, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-w', '--load', dest='load', type=str,
                        default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int,
                        default=1024, help='vector length')
    parser.add_argument('-d', '--dim', dest='dim', type=int,
                        default=7, help='number of point dimensions')
    parser.add_argument('-c', '--cluster', dest='cluster', type=int,
                        default=20, help='number of clusters')
    parser.add_argument('-t', '--test', dest='test', action='store_true', default=False,
                        help='testing phase')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning-rate')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        data_path = os.environ['data'] + "/2016_scivis_fpm/0.44/"
    except KeyError:
        data_path = './data/'

    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = SSAE(args.dim,args.cluster).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load:
        load_dict = torch.load(args.load, map_location=torch.device("cuda"))
        center = load_dict["center"]
        state = load_dict["state"]
        # print(state['enc.cs1.conv1.conv.weight'][0])
        # print(model.enc.cs1.conv1.conv.weight[0])
        model.set_center(center)
        model.load_state_dict(state)
        # print(model.enc.cs1.conv1.conv.weight[0])
        print('Model loaded from {}'.format(args.load))
        # exit()
    else:
        fpm = FPM(data_path+'/run01/003.vtu')
        r = torch.rand((5000,))
        r_idx = (r * len(fpm)).long()
        sample = wrap_data(fpm[r_idx].cuda())
        model.init_center(sample)

    if args.test:
        test(os.path.join(data_path,'run01/020.vtu'))
    else:
        train()



    ################################
    # t1 = time.time()
    # z,y_cls,y_rec = model(sample)
    # model.loss(sample[0],z,y_cls,y_rec)
    # print(time.time()-t1)
    ################################
    # loader = Loader(data_file,args.batch_size)
    # for epoch in range(1, args.epochs + 1):
    #     train(epoch)
    #     test(epoch)