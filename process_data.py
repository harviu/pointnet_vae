import random
import os
import sys
import math
import time

from vtkmodules.all import *
from vtkmodules.util import numpy_support
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.ckdtree import cKDTree,cKDTreeNode
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset
from yt.utilities.sdf import SDFRead
from thingking import loadtxt

    

class PointData(Dataset):
    def __init__(self,file_name,args,sampler="all"):
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
            if args.have_label:
                # read halo file if have label
                have_positive = True
                dir_name, real_file_name = os.path.split(file_name)
                timestep = int(real_file_name[-4:-2])
                if timestep == 0: 
                    timestep = 100
                try:
                    halo_file_name = dir_name+"/../rockstar/out_{}.list".format(timestep-2)
                    x,y,z,r = halo_reader(halo_file_name)
                    #normalize the halo positions
                    x = (x-mean[0])/std[0]
                    y = (y-mean[1])/std[1]
                    z = (z-mean[2])/std[2]
                    r /= std[2]
                except ValueError:
                    have_positive = False
        data = normalize(data,mean,std)
        coord = data[:,:3]
        kd = cKDTree(coord,leafsize=100)
        if args.have_label:
            if have_positive:
                nn = []
                try:
                    halo_num = len(x)
                except TypeError:
                    x = [x]
                    y = [y]
                    z = [z]
                    r = [r]
                for i in range(len(x)):
                    nn.append(kd.query_ball_point((x[i],y[i],z[i]),r[i]))
                positive = np.unique(np.concatenate(nn))
            else:
                positive = np.zeros((0),dtype=np.int)
            if sampler=="all":
                sample_id = np.arange(len(data))
                self.label = np.zeros((len(sample_id)),dtype=np.int64)
                self.label[positive] = 1
            elif sampler=="partial":
                all_sample = np.arange(len(data))
                negative = np.delete(all_sample,positive)
                if len(positive) > args.sample_size//2:
                    positive = np.random.choice(positive,args.sample_size//2,replace=False)
                    negative = np.random.choice(negative,args.sample_size//2,replace=False)
                    sample_id = np.concatenate((positive,negative)).astype(np.int)
                else:
                    num_negative = args.sample_size - len(positive)
                    negative = np.random.choice(negative,num_negative,replace=False)
                    sample_id = np.concatenate((positive,negative)).astype(np.int)
                self.label = np.zeros((len(sample_id)),dtype=np.int64)
                self.label[:len(positive)] = 1
        else:
            if sampler=="all":
                sample_id = np.arange(len(data))
            elif sampler=="partial":
                attr = data[:,3:]
                leaf_size = len(data)//args.sample_size
                # for some input files, enforce balanced tree will cause bug.
                attr_kd = cKDTree(attr,leaf_size,balanced_tree=False)
                leaf = all_leaf_nodes(attr_kd.tree)
                r = np.random.rand((len(leaf)))
                idx = []
                for i,l in enumerate(leaf):
                    indices = l.indices
                    idx.append(indices[int(len(indices)*r[i])])
                sample_id = idx

        if mode == "ball":
            self.nn = kd.query_ball_point(data[sample_id,:3],r,n_jobs=-1)
        elif mode == "knn":
            _, self.nn = kd.query(data[sample_id,:3],k,n_jobs=-1)
        self.data = data
        self.sample_id = sample_id
        self.k = int(k)
        self.r = r
        self.mode = mode
        self.have_label = args.have_label

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
            pc[:,:3] -= center[:3]
            #padding
            if len(nn_id) < self.k:
                dim = pc.shape[1]
                remaining = np.zeros((self.k-len(nn_id),dim))
                pc = np.concatenate((pc,remaining),axis=0)
                pc_length = len(nn_id)
            if self.have_label:
                return pc, pc_length, self.label[index]
            else:
                return pc, pc_length
        elif self.mode =="knn":
            pc = self.data[nn_id]
            center = self.data[nn_id[0]]
            pc[:,:3] -= center[:3]
            if self.have_label:
                return pc, self.label[index]
            else:
                return pc
    def __len__(self):
        return len(self.nn)

def IoU(predict,target):
    assert len(predict) == len(target)
    predict = np.array(predict)
    target = np.array(target)
    union = np.logical_or(predict,target)
    inter = np.logical_and(predict,target)
    return np.sum(inter)/np.sum(union)

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

def halo_reader(filename):
    ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
        loadtxt(filename, unpack=True)
    return x,y,z,Rvir/1000



def halo_writer(x,y,z,Rvir,outputname):
    haloData = vtkAppendPolyData()
    for i in range(len(x)):
        print(i,"/",len(x),end='\r')
        s = vtkSphereSource()
        s.SetCenter(x[i],y[i],z[i])
        s.SetRadius(Rvir[i])
        s.Update()
        input1 = vtkPolyData()
        input1.ShallowCopy(s.GetOutput())
        haloData.AddInputData(input1)
    haloData.Update()
    writer = vtkXMLPolyDataWriter()
    writer.SetInputConnection(haloData.GetOutputPort())
    writer.SetFileName(outputname)
    writer.Write()

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

def numpy_to_vtk(position:np.array,array_dict:dict):
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
    return data_save

def vtk_write(data_save,filename:str):
    writer = vtkXMLDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data_save)
    writer.Write()
