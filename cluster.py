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

    data_directory = data_path+"/2016_scivis_fpm/0.44/run41/024.vtu"
    # data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
    # data_directory = os.path.join("./data/020.vtu")
    state_dict = torch.load("states/CP1.pth")
    # state_dict = torch.load("states_saved/cos_label_knn128_dim10_vec512_CP35.pth")
    state = state_dict['state']
    args = state_dict['config']
    print(args)
    # halo_info = halo_reader(data_path+"/ds14_scivis_0128/rockstar/out_47.list")
    data = vtk_reader(data_directory)
    # pd = PointData(data,args,np.arange(len(data)))
    # # label = pd.label
    # model = AE(args).float().cuda()
    # model.load_state_dict(state)
    # latent = inference(pd,model,1500,args)
    # torch.save(latent,"latent")
    # torch.save(predict,"predict")
    # print(latent.shape)

    ################# kmeans ##################
    latent = torch.load("latent")
    print(latent.shape)
    # pca = PCA(5)
    # pca.fit(latent)
    # with open("pca","wb") as f:
    #     pickle.dump(pca,f)
    # with open("pca","rb") as f:
    #     pca = pickle.load(f)
    # p = pca.transform(latent)
    # print(p.shape)
    # predict = torch.load("predict")
    # predict = np.argmax(predict,1)
    # print(IoU(predict,label))
    # # data = vtk_reader(data_directory)
    # data = sdf_reader(data_directory)


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

    # pca = PCA(64)
    # latent = pca.fit_transform(latent)
    km = KMeans(5,n_init=10,n_jobs=-1)
    res = km.fit_predict(latent)
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


    array_dict = {
        "cluster": res,
        # "phi":data[:,-1],
        # "velocity":data[:,3:6],
        # "acceleration":data[:,6:9],
        "concentration": data[:,3],
        "velocity": data[:,4:]
    }
    vtk_data = numpy_to_vtk(data[:,:3],array_dict)

    # show(vtk_data)
    vtk_write(vtk_data,"test.vtu")


    ############### parallel coordinates #############
    # lat = np.concatenate((latent,res[:,None]),1)
    # df = pandas.DataFrame(data=lat[::100])
    # pandas.plotting.parallel_coordinates(df,class_column=16,color=('red', 'green', 'blue','yellow'))
    # plt.show()

    ################ tsne #################
    # tsn = TSNE(2)
    # d_latent = tsn.fit_transform(latent)
    # np.save("tsne_latent",d_latent)
    # d_latent = np.load("tsne_latent.npy")
    # plt.scatter(d_latent[:,0],d_latent[:,1],c = res,marker='.')
    # plt.show()


    # track_run(data_path+"/run05/",10,30,1,(-0.5, -2, 9),0.5,10,model,device,args.dim,True)
    # track_run(data_path+"/run05/",10,30,1,(1.2, 3.3, 9),0.5,20,model,device,args.dim,False)
    # track_run(data_path+"/run41/",15,20,1,(2, -1.2, 8.5),0.5,20,model,device,args.dim,False)
    # track_run(data_path+"/run41/",15,20,1,(3.7, 0.7, 7.9),0.5,10,model,device,args.dim,True)
    #best case

    # print("with")
    # track_run(data_path+"/run41/",10,30,1,(3.7, 0.7, 9.1),0.5,2,model,device,args.dim,True)
    # track_run(data_path+"/run05/",20,30,1,(-0.5, -3.7, 6.5),0.5,10,model,device,args.dim,True)
    # print("without")
    # track_run(data_path+"/run41/",10,21,1,(3.7, 0.7, 9.1),0.5,20,model,device,args.dim,False)
    # track_run(data_path+"/run05/",20,30,1,(-0.5, -3.7, 6.5),0.5,1000,model,device,args.dim,False)

    # track_run(data_path+"/run40/",10,20,1,(1, -3.4, 9.2),0.5,10,model,device,args.dim,True)
    # track_run(data_path+"/run32/",20,30,1,(3.7, 0.5, 6.5),0.5,10,model,device,args.dim,True)
    # track_run(data_path+"/run05/",20,30,1,(4.2, -2.1, 7),0.5,10,model,device,args.dim,True)
    # track_run(data_path+"/run05/",15,30,1,(4.2, -2.1, 8.2),0.5,1000,model,device,args.dim,False)

    # vtk_data = data_reader(data_path+"/run01/020.vtu")
    # # print(vtk_data)
    # numpy_data = data_to_numpy(vtk_data)
    # print(numpy_data.shape)
    # kdtree = KDTree(numpy_data[:,:3],1000)
    # x = np.arange(-5,5,0.5)
    # y = np.arange(-5,5,0.5)
    # z = np.arange(0,10,0.5)
    # index = []
    # for xx in x:
    #     for yy in y:
    #         for zz in z:
    #             if (xx**2 + yy**2 <25):
    #                 index.append((xx,yy,zz))

    
    # points = kdtree.query_ball_point((-0.9,-1.2,6.4),0.8)
    # # print(points)
    # numpy_data = numpy_data[:,:4]
    # index = numpy_data[:,:3][points]
    # pc = numpy_data[points]  
    # points = kdtree.query_ball_point(index,0.7)
    # data = []
    # for p in points:
    #     d = numpy_data[p]
    #     d = mean_sub(d)
    #     d[:,3] = (d[:,3] - 23.946)/55.08
    #     data.append(d)
    # data = to_tensor_list(data,device)
    # with torch.no_grad():
    #     latent = model.encode(data)
    #     latent = latent.cpu().numpy()

    # sc0 = np.concatenate((index,latent[:,0][:,None]),axis=1)
    # sc1 = np.concatenate((index,latent[:,1][:,None]),axis=1)
    # # print(sc.shape)
    # scatter_3d(pc)
    # scatter_3d(sc0)
    # scatter_3d(sc1)
    
        
    ############# latent shift
    
    # with open("data/latent_024","rb") as file:
    #     l1 = pickle.load(file)

    # with open("data/latent_025","rb") as file:
    #     l2 = pickle.load(file)

    # pca = PCA()
    # pca.fit(np.concatenate((l1,l2),axis = 0))
    
    # center = (1.5,-1,6.25)
    # di1 = data_path+"\\run41\\024.vtu"
    # di2 = data_path+"\\run41\\011.vtu"

    # data = data_reader(di1)
    # data = data_to_numpy(data)
    # data = data[:,:4]
    # scatter_3d(data,50,350,threshold=50)

    # data2 = data_reader(di2)
    # data2 = data_to_numpy(data2)
    # data2 = data2[:,:4]

    # start_df = latent_df(data,3,center,0.7,30,None,model,device,args.dim)
    # m = start_df.near_pc
    # pc1 = m.copy()
    # pc1 = mean_sub(pc1)
    # scatter_3d(pc1,None,None)

    # center2 = (0,0,7)

    # target = latent_df(data,3,center2,0.7,30,None,model,device,args.dim)
    # pc2 = target.near_pc.copy()
    # pc2 = mean_sub(pc2)
    # scatter_3d(pc2)

    # ms = mean_shift(m,target,ite=30)
    # ms.shift()
    # pc3 = target.near_pc.copy()
    # pc3 = mean_sub(pc3)
    # scatter_3d(pc3)

    # # center = target.center

    # print("original distance:",nn_distance(pc1,pc2))
    # print("after meanshift:",nn_distance(pc1,pc3))




    ######################################

    # target = p2[118901][None,:]
    # distance = p2-target
    # distance = torch.norm(distance, dim = -1)
    # print(distance.shape)

    # torch.save(distance,"to_self")

    # p = torch.load("to_self",map_location=device)

    # p = (p-min(p))/(max(p)-min(p)) * 357.19
    # data = data_reader(os.environ['data']+"/2016_scivis_fpm/0.44/run01/013.vtu")
    # new_concen = numpy_support.numpy_to_vtk(p)
    # data.GetPointData().GetArray(0).SetArray(new_concen,new_concen.GetSize(),1)
    # data.GetPointData().GetArray(0)
    # # plt.hist(p,bins=100)
    # # plt.show()
    # # print(len(p))
    # # print(data)

    # # print(min(p),max(p))
    # writer = vtkXMLUnstructuredGridWriter()
    # writer.SetFileName("to_self.vtu")
    # writer.SetInputData(data)
    # writer.Write()

    # sort = np.argsort(p)
    # print(sort)

    # data = data_reader(data_path + "/0.44/run01/013.vtu")
    # coord = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    # print(coord[118901])
    # data = data_reader(data_path + "/0.44/run01/010.vtu")
    # coord = data.GetPoints().GetData()
    # xyz = coord.GetTuple(118901)
    # print(xyz)

    # data = data_reader("new.vtu")
    # concen = numpy_support.vtk_to_numpy(data.GetPointData().GetArray(0))
    # print(np.where(concen>350))
    # show("to_self.vtu")
    # show(data_path + "/0.44/run01/010.vtu")

    # data_directory = os.path.join(data_path,"run41/015.vtu")

    # data = data_reader(data_directory)
    # data = data_to_numpy(data)
    # print(data.shape)


