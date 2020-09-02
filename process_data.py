import random
import os
import sys
import math
import time

from vtk import *
from vtkmodules.util import numpy_support
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.ckdtree import cKDTree,cKDTreeNode
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset
from yt.utilities.sdf import SDFRead

all_sampler = lambda data:np.arange(len(data))
def balance_sampler(data,num):
    attr = data[:,3:]
    leaf_size = len(data)//num
    # for some input files, enforce balanced tree will cause bug.
    kd = cKDTree(attr,leaf_size,balanced_tree=False)
    leaf = all_leaf_nodes(kd.tree)
    r = np.random.rand((len(leaf)))
    idx = []
    for i,l in enumerate(leaf):
        indices = l.indices
        idx.append(indices[int(len(indices)*r[i])])
    return idx
    

class PointData(Dataset):
    def __init__(self,file_name,args,sampler=all_sampler):
        source = args.source
        mode = args.mode
        k = args.k
        r = args.r
        if source == "fpm":
            data = vtk_reader(file_name)
            mean = [0, 0, 5, 23.9, 0, 0, 0.034]
            std = [2.68, 2.68, 3.09, 55.08, 0.3246, 0.3233, 0.6973]
        elif source == "cos":
            data = sdf_reader(file_name)
            mean = [30.4, 32.8, 32.58, 0, 0, 0, 0, 0, 0, -732720]
            std = [18.767, 16.76, 17.62, 197.9, 247.2, 193.54, 420.92, 429, 422.3, 888474]
        data = normalize(data,mean,std)
        coord = data[:,:3]
        kd = cKDTree(coord,leafsize=100)
        # calculate the samples acoording to sampler function.
        sample_id = sampler(data)
        if mode == "ball":
            self.nn = kd.query_ball_point(data[sample_id,:3],r,n_jobs=-1)
        elif mode == "knn":
            _, self.nn = kd.query(data[sample_id,:3],k,n_jobs=-1)
        self.data = data
        self.sample_id = sample_id
        self.k = int(k)
        self.r = r
        self.mode = mode

    def __getitem__(self, index):
        nn_id = self.nn[index]
        if self.mode == "ball":
            #cutting
            if len(nn_id) >= self.k:
                nn_id = nn_id[:self.k]
                pc_length = self.k
            #point cloud and center point
            pc = self.data[nn_id]
            center = self.data[nn_id[0]]
            pc -= center
            #padding
            if len(nn_id) < self.k:
                dim = pc.shape[1]
                remaining = np.zeros((self.k-len(nn_id),dim))
                pc = np.concatenate((pc,remaining),axis=0)
                pc_length = len(nn_id)
            return pc, pc_length
        elif self.mode =="knn":
            pc = self.data[nn_id]
            center = self.data[nn_id[0]]
            pc -= center
            return pc
    def __len__(self):
        return len(self.nn)

def scatter_3d(array,vmin=None,vmax=None,threshold = -1e10,center=None,save=False,fname=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    array = array[array[:,3] > threshold]
    ax.scatter(array[:,0],array[:,1],array[:,2],c=array[:,3],marker='.',vmin=vmin,vmax=vmax)
    if center is not None:
        ax.scatter(center[0],center[1],center[2],c="red",marker='o')
    # ax2 = fig.add_subplot(122,projection='3d',sharex=ax,sharey=ax,sharez=ax)
    # ax2.scatter(array2[:,0],array2[:,1],array2[:,2],c=array2[:,3],marker='^',vmin=-1,vmax=1)
    if save:
        plt.savefig(fname)
    else:
        plt.show()

def vtk_reader(filename):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))[:,None]
    velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
    point_data = np.concatenate((coord,concen,velocity),axis=-1)
    return point_data

def sdf_reader(filename):
    particles = SDFRead(filename)
    h_100 = particles.parameters['h_100']
    width = particles.parameters['L0']
    cosmo_a = particles.parameters['a']
    kpc_to_Mpc = 1./1000
    convert_to_cMpc = lambda proper: (proper ) * h_100 * kpc_to_Mpc / cosmo_a + 31.25
    numpy_data = np.array(list(particles.values())[2:-1]).T
    numpy_data[:,:3] = convert_to_cMpc(numpy_data[:,:3])
    return numpy_data
    
def normalize(data,mean,std):
    return (data - mean)/std

def all_leaf_nodes(node):
    if node.greater==None and node.lesser==None:
        return [node]
    else:
        return(all_leaf_nodes(node.lesser)+all_leaf_nodes(node.greater))

def collect_file(directory,mode="fpm",shuffle=False):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if mode == "fpm":
                if filename.endswith(".vtu") and filename != "000.vtu":
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
            if mode == "cos":
                if "ds14" in filename.split('_'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
    if shuffle:
        random.shuffle(file_list)
    return file_list
    
def min_max(file_list,mode="fpm"):
    all_min = None
    all_max = None
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = vtk_reader(f) if mode == "fpm" else sdf_reader(f)
        f_min = np.min(data,axis=0)
        f_max = np.max(data,axis=0)
        if all_min is None:
            all_min = f_min
            all_max = f_max
            total = np.zeros((data.shape[1]))
            length = 0
        else:
            all_min = np.where(all_min < f_min,all_min,f_min)
            all_max = np.where(all_max > f_max,all_max,f_max)
        total += np.sum(data,axis = 0)
        length += len(data)
    mean = total/length
    print("mean: ", mean)
    print("min: ", all_min)
    print("max: ", all_max)
    return mean,all_min,all_max

def std(file_list,mean,mode="fpm"):
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = vtk_reader(f) if mode == "fpm" else sdf_reader(f)
        if i==0:
            total = np.zeros((data.shape[1]))
            length = 0
        data = (data - mean) ** 2
        total += np.sum(data,axis = 0)
        length += len(data)
    std = np.sqrt(total/length)
    print("std: ", std)
    return std

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