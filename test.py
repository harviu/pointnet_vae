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
from multiprocessing import Pool

import numpy as np

from model import VAE
from process_data import *
from vtkmodules.all import *

from scipy.spatial.ckdtree import cKDTree
import os
import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
from annoy import AnnoyIndex
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from thingking import loadtxt
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors

def vtk_write(position:np.array,array_dict:dict,filename:str):
    vtk_position = numpy_support.numpy_to_vtk(position)
    points = vtkPoints()
    points.SetData(vtk_position)
    data_save = vtkUnstructuredGrid()
    data_save.SetPoints(points)
    pd = data_save.GetPointData()
    for k, v in array_dict.items():
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        pd.AddArray(vtk_array)
    writer = vtkXMLDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data_save)
    writer.Write()


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def func(d):
    return d[1].query_ball_point(d[0],0.7)

if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    try:
        data_path = os.environ['data'] + "/2016_scivis_fpm/0.44/"
    except KeyError:
        data_path = './data/'

    # data_directory = os.path.join(data_path,"run41/030.vtu")
    # data_directory = os.path.join("./data/020.vtu")
    # data = vtk_reader(data_directory)
    # print(data.shape)
    # fpm = FPM(data_directory,"ball")
    # loader = DataLoader(fpm, batch_size=32, shuffle=True, drop_last=False)
    # for i in loader:
    #     print(i.shape)
    # data_directory = os.environ['data'] + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
    # data = sdf_reader(data_directory)
    # data = normalize(data,3,10)
    
    # ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
    #     loadtxt(os.environ['data']+"/ds14_scivis_0128/rockstar/out_{:d}.list".format(49-2), unpack=True)
    # center = np.array((x,y,z)).T
    # data_save = vtkUnstructuredGrid()
    # data_save.SetPoints(points)
    # print(data_save)


    # attr = torch.load("results/new_model/latent")
    # km = KMeans(4,n_init=10)
    # res = km.fit_predict(attr)
    # np.save("results/new_model/kmean_result",res)
    # res = np.load("kmean_result.npy")

    # numpy_data = data

    ###############################
    # t1 = time.time()
    # kd = NearestNeighbors(256,algorithm="kd_tree",leaf_size=30,n_jobs=8)
    # kd.fit(data[:,:3])
    # kd = cKDTree(data[:,:3],30)
    # for i in range(int(200/20)):
    #     t = torch.rand((1,7,20000),dtype=torch.float32,device="cpu")
    #     knn(t,256)
    # kd = AnnoyIndex(3,'euclidean')
    # for i in range(len(data)):
    #     kd.add_item(i,data[i,:3])
    # kd.build(1)
        
    # print(time.time()-t1)
    ################################
    ################################
    # t1 = time.time()
    # for i in range(len(data)):
    #     kd.get_nns_by_item(i,256)
    # # pool = Pool(8)
    # # idx = pool.map(func,zip(data[:,:3],[kd]*len(data)))
    # knn = kd.query_ball_point(data[:,:3],0.7,n_jobs=16)
    # print(time.time()-t1)
    # print(knn)
    ################################
    # np.save("01200",knn[1])

    # idx = np.load("./data/new.npy")
    # print(idx[0,:,7:])

    ################# kmeans ##################
    # latent = torch.load("results/new_model/latent")
    # data = data[:len(latent)]
    # km = KMeans(4,n_init=10)
    # res = km.fit_predict(latent)
    # print(data.shape,res.shape)
    # print(np.max(res))

    # array_dict = {
    #     "cluster": res,
    #     "concentration":data[:,3],
    #     "velocity":data[:,4:],
    # }

    # vtk_write(data[:,:3],array_dict,"results/new_model/030_cluster.vtu")

    # cluster = []
    # for i in range(4):
    #     cluster.append(data[res==i,:])
    #     scatter_3d(cluster[i])


    # cluster_show = cluster[0]
    # sli = np.logical_and(cluster_show[:,2]>7.9, cluster_show[:,2]<8.0)
    # # scatter_3d(cluster_show)
    # cluster_show = cluster_show[sli]
    # plt.scatter(cluster_show[:,0],cluster_show[:,1],c="red")

    # cluster_show = cluster[2]
    # sli = np.logical_and(cluster_show[:,2]>8.9, cluster_show[:,2]<9.0)
    # # scatter_3d(cluster_show)
    # cluster_show = cluster_show[sli]
    # plt.scatter(cluster_show[:,0],cluster_show[:,1],c="blue")

    # cluster_show = cluster[3]
    # sli = np.logical_and(cluster_show[:,2]>7.9, cluster_show[:,2]<8.0)
    # # scatter_3d(cluster_show)
    # cluster_show = cluster_show[sli]
    # plt.scatter(cluster_show[:,0],cluster_show[:,1],c="green")
    
    # plt.show()


    ############### parallel coordinates #############
    # lat = np.concatenate((latent,res[:,None]),1)
    # df = pd.DataFrame(data=lat[::100])
    # pd.plotting.parallel_coordinates(df,class_column=16,color=('red', 'green', 'blue','yellow'))
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

    ############# guided shift
    # first test this on the same frame
    # center = (1.5,-1,6.25)
    # data = data_reader(data_path+r"\run41\024.vtu")
    # data = data_to_numpy(data)
    # data = data[:,:args.dim]
    # df = data_frame(data,3,center,0.7,bins=1000)
    # target_pc = df.near_pc.copy()

    # # # set the start center
    # df.center = (2,-0.5,6.55)
    # df.update()
    # guide = LatentMax(model,target_pc,device,args.dim)
    # gs = guided_shift(target_pc,df,guide)
    # gs.shift()


    ############################## convert one file to new features
    # filename = data_path + "/run41/024.vtu"
    # data = data_reader(filename)
    # data = data_to_numpy(data)
    # coord = data[:,:3]
    # attr = data[:,3:]
    # mean=[2.39460057e+01, -4.29336209e-03, 9.68809421e-04, 3.44706680e-02]
    # std=[55.08245731,  0.32457581,  0.32332313,  0.6972805]
    # data[:,3:] = (data[:,3:] - mean)/std
    # coord_kd = KDTree(coord)
    # i = 0
    # dd = []
    # for point in coord:
    #     ball = coord_kd.query_ball_point(point,r=0.7)
    #     print("{}/{}".format(i+1,len(data)),end='\r')
    #     dd.append(data[ball])
    #     i+=1
    # with open("run41_025","wb") as file:
    #     pickle.dump(dd,file)

    ################## encode to latent ##############
    # with open("run41_024","rb") as file:
    #     data = pickle.load(file)
    #     data = to_tensor_list(data,device,args.dim)

    # model.eval()
    # latent = torch.zeros((len(data),args.vector_length))
    # with torch.no_grad():
    #     for i in range(0,len(data),args.batch_size):
    #         batch = data[i:i+args.batch_size]
    #         latent[i:i+args.batch_size] = model.encode(batch)
    #         print("{}/{}".format(i+1,len(data)),end='\r')

    # with open("latent","wb") as file:
    #     pickle.dump(latent,file)
    # print(latent.shape)

    # with open("data/latent_024","rb") as file:
    #     d = pickle.load(file).cpu()

    # pca = PCA(n_components=5)
    # d_embedded = pca.fit_transform(d)
    # print(pca.explained_variance_ratio_)
    # pc = np.concatenate((coord,d_embedded[:,3:]),axis=1)
    # print(pc.shape)
    # scatter_3d(pc[::100],None,None)
    # scatter_3d(data[::100],None,None)

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


