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


def kmean_and_save(data,attr,k=10,mode="fpm"):
    km = KMeans(k,n_init=10,n_jobs=-1)
    res = km.fit_predict(attr)
    # if mode=="cos":
    #     array_dict = {
    #         "cluster": res,
    #         "phi":data[:,-1],
    #         "velocity":data[:,3:6],
    #         "acceleration":data[:,6:9],
    #     }
    # else:
    #     array_dict = {
    #         "cluster": res,
    #         "concentration": data[:,3],
    #         "velocity": data[:,4:]
    #     }
    # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_write(vtk_data,"cos_latent_last49.vtu")
    return res


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


if __name__ == "__main__":
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    mode = "fpm"

    if mode == "fpm":
        data_directory = data_path+"/2016_scivis_fpm/0.44/run41/025.vtu"
        state_dict_directory = "states_saved/fpm_knn128_dim7_vec64_CP5.pth"
        data = vtk_reader(data_directory)
    else:
        data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
        state_dict_directory = "states_saved/cos_label_knn128_dim10_vec512_CP35.pth"
        halo_directory = data_path + '/ds14_scivis_0128/rockstar/out_47.list'
        data = sdf_reader(data_directory)
    
    state_dict = torch.load(state_dict_directory)
    state = state_dict['state']
    args = state_dict['config']
    print(args)
    model = AE(args).float().cuda()
    model.load_state_dict(state)
    if args.have_label:
        hp = halo_reader(halo_directory)
        pd = PointData(data,args,np.arange(len(data)),hp)
        latent,predict = inference(pd,model,1500,args)
        label = pd.label
        # torch.save(latent,"cos_latent_middle49")
        # torch.save(predict,"predict")
    else:
        pd = PointData(data,args,np.arange(len(data)))
        latent = inference(pd,model,1500,args)
        # torch.save(latent,"fpm_latent_25")
    

    ################# analysis ##################
    # predict = torch.load("results/cos_label/predict49")
    # predict = np.argmax(predict,1)
    # print(IoU(predict,label))
    # latent = torch.load("results/cos_label/cos_latent_last49")

    # pca = PCA(16)
    # latent = pca.fit_transform(latent)
    res = kmean_and_save(data,latent,4)

    ############### pca #############
    pca = PCA(4)
    pca.fit(latent)
    with open("pca","wb") as f:
        pickle.dump(pca,f)
    # with open("pca","rb") as f:
    #     pca = pickle.load(f)
    # p = pca.transform(latent)

    ############### DBSCAN #############

    # data = data
    # res = np.zeros((len(data)),dtype=np.int)
    # res[save_idx] = 1
    # data = data[save_idx]
    # np.save("interesting_cluster",data)
    # data = np.load("interesting_cluster.npy")
    # print(data.shape)
    # db = DBSCAN(0.44,30)
    # res2 = db.fit_predict(data[:,:3])
    # res2 = res2.astype(np.long)
    # print(res2.shape)


    ############### parallel coordinates #############
    lat = np.concatenate((latent,res[:,None]),1)
    df = pandas.DataFrame(data=lat[::100])
    pandas.plotting.parallel_coordinates(df,class_column=args.vector_length,color=('red', 'green', 'blue','yellow'))
    plt.show()

    ################ tsne #################
    # tsn = TSNE(2)
    # d_latent = tsn.fit_transform(latent)
    # np.save("tsne_latent",d_latent)
    # d_latent = np.load("tsne_latent.npy")
    # plt.scatter(d_latent[:,0],d_latent[:,1],c = res,marker='.')
    # plt.show()

    #################### Hierarchical ####################
    # root = Node(np.arange(len(latent)))
    # hierk(root)
    # node_list = [root]
    # save_idx = []
    # idx = 0
    # while(len(node_list)>0):
    #     node = node_list.pop(0)
    #     if node is not None:
    #         print(idx,node.level)
    #         if node.level==5 and (idx==16 or idx==18):
    #             save_idx+=list(node.value)
    #         # sub_data = data[node.value]
    #         # array_dict = {
    #         #     "concentration":sub_data[:,3],
    #         #     "velocity":sub_data[:,4:],
    #         # }
    #         # vtk_data = numpy_to_vtk(sub_data[:,:3],array_dict)
    #         # show(vtk_data,outfile="{}_{}".format(node.level,idx))
    #         node.idx = idx
    #         node_list.append(node.left)
    #         node_list.append(node.right)
    #     idx += 1
    # # traverse(root)