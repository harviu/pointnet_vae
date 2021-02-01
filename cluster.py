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


if __name__ == "__main__":
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'
    mode = "cos"
    # IoU_list = []
    # loss_list = []
    # for i in range(2,100,1):
    #     print(i)
    i = 49
    if mode == "fpm":
        data_directory = data_path+"/2016_scivis_fpm/0.44/run41/025.vtu"
        # state_dict_directory = "states_saved/fpm_knn128_dim7_vec64_CP35.pth"
        state_dict_directory = "states_saved/fpm_k128_v64/CP35.pth"
        data = vtk_reader(data_directory)
    else:
        if i == 100:
            data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.{:02d}00'.format(i)
        else:
            data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.{:02d}00'.format(i)
        state_dict_directory = "states_saved/cos_k64_v256/CP29.pth"
        # state_dict_directory = "states_saved/cos_k128_v256/CP16.pth"
        # state_dict_directory = "states_saved/cos_k128_v512/CP35.pth"
        # state_dict_directory = "states_saved/cos_k128_v768/CP13.pth"
        # state_dict_directory = "states_saved/cos_k256_v512/CP20.pth"
        # state_dict_directory = "states_saved/cos_k64_v512/CP35.pth"
        # state_dict_directory = "states_saved/cos_k32_v256/CP35.pth"
        halo_directory = data_path + '/ds14_scivis_0128/rockstar/out_{}.list'.format(i-2)
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
        latent,predict,loss = inference(pd,model,1000,args)
        label = pd.label
        # torch.save(latent,"cos_latent_middle49")
        # torch.save(predict,"predict")
    else:
        pd = PointData(data,args,np.arange(len(data)))
        latent = inference(pd,model,1500,args)
        # torch.save(latent,"fpm_latent_25")

    

    ################# analysis ##################
    predict = np.argmax(predict,1)
    IoU_value = IoU(predict,label)
    sub = predict + label
    # IoU_list.append(IoU_value)
    # loss_list.append(loss)
    # np.save("loss_list",loss_list)
    # np.save("iou_list",IoU_list)


    # latent = torch.load("results/cos_label/cos_latent_last49")

    # pca = PCA(4)
    # pca.fit(latent)
    # pickle.dump( pca, open( "states_saved/cos_k64_v256/pca_late", "wb" ) )

    # pca = PCA(16)
    # latent = pca.fit_transform(latent)
    # km = KMeans(4,n_init=3,n_jobs=-1)
    # embedding = km.fit_predict(latent)
    # new_cluster = km.cluster_centers_
    # print(km.cluster_centers_)

    if mode=="cos":
        array_dict = {
            "predict": predict,
            "label": label,
            "sub": sub,
            "phi":data[:,-1],
            "velocity":data[:,3:6],
            "acceleration":data[:,6:9],
        }
    else:
        array_dict = {
            # "pca": pca_output,
            # "mean": mean_neighbor,
            "embedding": embedding,
            "concentration": data[:,3],
            "velocity": data[:,4:]
        }
    vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    vtk_write(vtk_data,"cos_49.vtu")

    # hp = halo_reader(halo_directory)
    # print(hp)
    # halo_writer(hp[0],hp[1],"halo49.vtu")

    # cluster_centers = np.load("cluster_center.npy")[3]
    # distance = np.sum((new_cluster - cluster_centers) ** 2,-1)
    # interested_cluster = np.argmin(distance)
    # print(distance,interested_cluster)

    ############### DBSCAN #############
    # cluster_id = embedding
    # data = data[cluster_id==interested_cluster]
    # # print(data.shape)
    # db = DBSCAN(0.44,30)
    # res2 = db.fit_predict(data[:,:3])
    # res2 = res2.astype(np.int)
    # print(np.max(res2))

    # array_dict = {
    #     # "pca": pca_output,
    #     # "mean": mean_neighbor,
    #     "embedding": res2,
    #     "concentration": data[:,3],
    #     "velocity": data[:,4:]
    # }
    # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_write(vtk_data,"result_overview.vtu")


    ############### parallel coordinates #############
    # lat = np.concatenate((latent,embedding[:,None]),1)
    # df = pandas.DataFrame(data=lat[::100])
    # plt.figure(figsize=(9,4))
    # plt.xlabel('Embedding Dimension')
    # plt.ylabel('Value')
    # pandas.plotting.parallel_coordinates(df,class_column=args.vector_length,color=('#3985ad'))
    # legend = plt.legend()
    # legend.remove()
    # plt.show()

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

    ########## convert to density #############
    # coord = data[:,:3]
    # kd = cKDTree(coord,leafsize=100)
    # x = np.linspace(-5,5,64)
    # y = np.linspace(-5,5,64)
    # z = np.linspace(0,10,64)
    # xv,yv,zv = np.meshgrid(x,y,z)
    # xv = xv.reshape(-1)
    # yv = yv.reshape(-1)
    # zv = zv.reshape(-1)
    # idx = np.array([xv,yv,zv]).T
    # nn = kd.query_ball_point(idx,r=0.4,n_jobs=-1)
    # density = np.zeros((len(idx)))
    # print(density.shape)
    # for i,n in enumerate(nn):
    #     if len(n) > 1 :
    #         center = idx[i]
    #         n = data[n]
    #         dis = np.sqrt(np.sum((n[:,:3]-center[:3])**2,-1))
    #         weight = 1- 1/(dis + 1e-8)
    #         density[i] = np.average(n,0,weight)[3]
    #         print(i)
    # vtk_data = vtk.vtkImageData()
    # vtk_data.SetDimensions(64,64,64)
    # scaler = numpy_support.numpy_to_vtk(density)
    # vtk_data.GetPointData().AddArray(scaler)
    # writer = vtk.vtkXMLImageDataWriter()
    # writer.SetFileName("test.vti")
    # writer.SetInputData(vtk_data)
    # writer.Write()