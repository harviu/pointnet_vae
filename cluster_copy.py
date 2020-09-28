from process_data import *

import random
import os
import argparse
import pickle
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import AE
from process_data import *
from train import inference

from scipy.spatial.ckdtree import cKDTree
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
import pandas 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simple import show
from mean_shift import LatentRetriever


class Node():
    def __init__(self,value):
        self.left = None
        self.right = None
        self.value = value

def traverse(node:Node):
    idx = node.value
    res[idx] = node.idx
    if node.left is not None:
        traverse(node.left)
    if node.right is not None:
        traverse(node.right)
    
def hierk(node:Node,level=0):
    idx = node.value
    node.level=level
    if len(idx)<20000:
        return node
    else:
        km = KMeans(2,n_init=10,n_jobs=-1)
        res = km.fit_predict(latent[idx])
        n1 = Node(idx[res==0])
        n2 = Node(idx[res==1])
        node.left = hierk(n1,level+1)
        node.right = hierk(n2,level+1)
        return node


# if __name__ == "__main__":
try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data/'
mode = "fpm"
cluster_centers = np.load("cluster_center.npy")[3]

state_dict_directory = "states_saved/fpm_knn128_dim7_vec64_CP35.pth"
state_dict = torch.load(state_dict_directory)
state = state_dict['state']
args = state_dict['config']
print(args)
model = AE(args).float().cuda()
model.load_state_dict(state)

for d in os.listdir(data_path+"/2016_scivis_fpm/0.44/"):
    d = "run41"
    print(d)
    finger_numbers = []
    # if d == "run41" or d=="run01": 
    #     continue

    # get reference center
    # print("getting reference center")
    # data_directory = data_path+"/2016_scivis_fpm/0.44/{}/025.vtu".format(d)
    # data = vtk_reader(data_directory)
    # pd = PointData(data,args,np.arange(len(data)))
    # latent = inference(pd,model,1500,args)
    # km = KMeans(8,n_init=3,n_jobs=-1) ###
    # embedding = km.fit_predict(latent)
    # new_cluster = km.cluster_centers_
    # distance = np.sum((new_cluster - cluster_centers) ** 2,-1)
    # interested_cluster = np.argmin(distance)
    # print(distance,interested_cluster)
    # cluster_centers = new_cluster[interested_cluster]

    for i in range(0,121):
        print("time:",i)
        if mode == "fpm":
            data_directory = data_path+"/2016_scivis_fpm/0.44/{}/{:03d}.vtu".format(d,i)
            state_dict_directory = "states_saved/fpm_knn128_dim7_vec64_CP35.pth"
            data = vtk_reader(data_directory)
        else:
            data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
            state_dict_directory = "states_saved/cos_label_knn128_dim10_vec512_CP35.pth"
            halo_directory = data_path + '/ds14_scivis_0128/rockstar/out_47.list'
            data = sdf_reader(data_directory)
        
        # state_dict = torch.load(state_dict_directory)
        # state = state_dict['state']
        # args = state_dict['config']
        # print(args)
        # model = AE(args).float().cuda()
        # model.load_state_dict(state)

        # if args.have_label:
        #     hp = halo_reader(halo_directory)
        #     pd = PointData(data,args,np.arange(len(data)),hp)
        #     latent,predict = inference(pd,model,1500,args)
        #     label = pd.label
        #     # torch.save(latent,"cos_latent_middle49")
        #     # torch.save(predict,"predict")
        # else:
        #     pd = PointData(data,args,np.arange(len(data)))
        #     latent = inference(pd,model,1500,args)
        #     torch.save(latent,"run41_latent/{:03d}".format(i))

        

        ################# analysis ##################
        # predict = torch.load("results/cos_label/predict49")
        # predict = np.argmax(predict,1)
        # print(IoU(predict,label))
        # latent = torch.load("results/cos_label/cos_latent_last49")

        # pca = PCA(16)
        # latent = pca.fit_transform(latent)

        latent = torch.load("run41_latent/{:03d}".format(i))
        print(latent.shape)
        km = KMeans(5,n_init=3,n_jobs=-1)
        embedding = km.fit_predict(latent)
        new_cluster = km.cluster_centers_

        # print(km.cluster_centers_)

        # if mode=="cos":
        #     array_dict = {
        #         "embedding": embedding,
        #         "phi":data[:,-1],
        #         "velocity":data[:,3:6],
        #         "acceleration":data[:,6:9],
        #     }
        # else:
        #     array_dict = {
        #         # "pca": pca_output,
        #         # "mean": mean_neighbor,
        #         "embedding": embedding,
        #         "concentration": data[:,3],
        #         "velocity": data[:,4:]
        #     }
        # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
        # vtk_write(vtk_data,"test.vtu")

        distance = np.sum((new_cluster - cluster_centers) ** 2,-1)
        interested_cluster = np.argmin(distance)
        # print(distance,interested_cluster)

        ############### DBSCAN #############
        cluster_id = embedding
        data = data[cluster_id==interested_cluster]

        db = DBSCAN(0.44,64)
        res2 = db.fit_predict(data[:,:3])
        res2 = res2.astype(np.int)
        print(np.max(res2))
        finger_numbers.append(np.max(res2))

        array_dict = {
            # "pca": pca_output,
            # "mean": mean_neighbor,
            "embedding": res2,
            "concentration": data[:,3],
            "velocity": data[:,4:]
        }
        vtk_data = numpy_to_vtk(data[:,:3],array_dict)
        # show(vtk_data,outfile="{}".format(i),show=True)
        vtk_write(vtk_data,"run41/{:03d}.vtu".format(i))
    exit()
    # np.save("finger_numbers_{}".format(d),finger_numbers)
##########
