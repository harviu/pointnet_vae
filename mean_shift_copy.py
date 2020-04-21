import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pickle

from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from process_data import *
from latent_max import LatentMax

MAX = 360
eps = 1e-10

class data_frame():
    def __init__(self,data,n_channel,center,h,bins=None,ranges=None):
        self.data = data
        self.coord = data[:,:n_channel]
        self.attr = data[:,n_channel:]
        self.kd = KDTree(self.coord,1000)
        self.bins = bins
        self.h = h
        self.coord_dim = n_channel
        self.set_center(center)
        self.set_range(ranges)
        self.update_hist()

    def set_range(self,ranges):
        self.ranges = ranges
        if self.ranges is None:
            rmax = np.max(self.near_attr,axis=0)
            rmin = np.min(self.near_attr,axis=0)
            self.ranges = np.stack((rmin,rmax),axis=-1)

    def set_center(self,center):
        self.center = center
        self.near = self.kd.query_ball_point(self.center,self.h)
        self.near_coord = self.coord[self.near]
        self.near_attr = self.attr[self.near]
        self.near_pc = self.data[self.near]

    def update_hist(self):
        self.hist = weighted_hist(self.near_coord,self.near_attr, self.center,self.h,self.bins,self.ranges)


class latent_df(data_frame):
    def __init__(self,data,n_channel,center,h,bins,ranges,model,device,dim,pca=None):
        self.model = model
        self.device = device
        self.dim = dim
        self.pca = pca

        latent = np.zeros((len(data),2))
        self.latent_mask = [False]* len(data)
        self.data = np.concatenate((data,latent),axis=1)
        self.coord = self.data[:,:n_channel]
        self.attr = self.data[:,n_channel:]
        self.kd = KDTree(self.coord,1000)
        self.bins = bins
        self.h = h
        self.coord_dim = n_channel

        self.set_center(center)
        self.set_range(ranges)
        self.update_hist()

    def set_center(self, center):
        super().set_center(center)
        self.calc_latent(self.model,self.device,self.dim,self.pca)

    def calc_latent(self,model,device,dim,pca):
        """
        calculte the latent vectors of points around center
        """
        not_cal = []
        for n in self.near:
            if not self.latent_mask[n]:
                not_cal.append(n)
                self.latent_mask[n] = True
        coord = self.coord[not_cal]
        t = self.kd.query_ball_point(coord,0.7)
        if len(t) > 0:
            x = []
            for tt in t:
                x.append(self.data[tt][:,:-2])
            x = prepare_for_model(x,device,3,dim)
            # print(x[0].shape)
            with torch.no_grad():
                y = model.encode(x)
            latent = y.detach().cpu().numpy()   
            # print(latent.shape) 
            self.data[not_cal,-2:] = latent
            self.near_coord = self.coord[self.near]
            self.near_attr = self.attr[self.near]
            self.near_pc = self.data[self.near]


            
    
def get_latent(model,x:np.ndarray,device,dim):
    x = prepare_for_model([x],device,3,dim)
    # scatter_3d(x[0])
    with torch.no_grad():
        y = model.encode(x)
    return y.detach().cpu().numpy()        

def weighted_hist(near_coord,near_attr, center, h,bins,ranges):
    weights = 1 - np.sum(((near_coord-center)/h)**2,axis=-1)
    hist = np.histogramdd(
        near_attr,
        bins = bins,
        range = ranges,
        weights = weights,
        density = True)
    hist[0][hist[0]<0] = 0
    # print(hist[0])
    return hist[0]

class guided_shift():
    def __init__(self, target_pc, init_df: data_frame, guide: LatentMax):
        self.target_pc = target_pc
        self.init_df = init_df
        # guide is a latent optimizer
        self.guide = guide
        # init mean shifter
        self.ms = mean_shift(None,self.init_df,ite=30)
        # how about stopping criteria?

    def shift(self):
        while True:
            # update init point cloud
            init_pc = self.init_df.near_pc
            mean =  np.mean(init_pc[:,:3],axis=0)
            init_pc[:,:3] -= mean[None,:]
            # scatter_3d(init_pc)
            # copy = init_pc.copy()
            # get the next pc through latent optimizer
            # notice the init_pc is changed inplace
            next_pc = self.guide.take_one_step_to_target(init_pc)
            next_pc[:,:3] += mean[None,:]
            # print(next_pc)
            # mean shift to next pc
            self.ms.target = next_pc
            self.ms.shift()
            # when to stop?
            # break
        

class mean_shift():
    def __init__(self,target,data,ite=20,dis=0.01):
        """
        data: data_frame object to search
        target: target point cloud
        """
        self.target = target
        self.data = data
        self.ite = ite
        self.dis = dis

    def adaptive_range(self,data,target):
        coord_dim = self.data.coord_dim
        ### adaptive range 
        data_rmax = np.max(data.near_attr,axis=0)
        data_rmin = np.min(data.near_attr,axis=0)
        target_rmax = np.max(target[:,coord_dim:],axis=0)
        target_rmin = np.min(target[:,coord_dim:],axis=0)
        new_rmin = np.where(data_rmin<target_rmin,data_rmin,target_rmin)
        new_rmax = np.where(data_rmax>target_rmax,data_rmax,target_rmax)
        new_ranges = np.stack((new_rmin,new_rmax),axis=-1)
        # new_ranges = ((0,MAX),(-1,1),(-1,1),(-1,1))

        # set new range
        data.set_range(new_ranges)
        data.update_hist()

        target_center = np.mean(target[:,:coord_dim],axis=0)
        target_hist = weighted_hist(target[:,:coord_dim],target[:,coord_dim:],target_center,data.h,data.bins,new_ranges)

        return target_hist, new_ranges

    def next_center(self):

        data = self.data
        target = self.target
        coord_dim = self.data.coord_dim
        
        target_hist, new_ranges = self.adaptive_range(data,target)

        weights = np.sqrt(target_hist/(data.hist+eps))

        near_bins = self._get_bins(data.near_attr,new_ranges,data.bins)
        
        new_center = np.zeros((len(data.center),))
        w_sum = eps
        near_w = np.zeros((len(near_bins)))
        for i in range(len(near_bins)):
            b = near_bins[i]
            w = weights[tuple(b)]
            new_center += w * data.near_coord[i]
            w_sum += w
            near_w[i]=w
        # print(near_w)

        new_center /= w_sum
        
        # print(new_center)
        return new_center

    def _get_bins(self,samples, ranges, bins):
        n_dims = samples.shape[1]
        sample_bins = np.zeros_like(samples,dtype=np.int)
        for d in range(n_dims):
            if ranges is None:
                max = np.max(samples[:,d])
                min = np.min(samples[:,d])
            else:
                max = ranges[d][1]
                min = ranges[d][0]
            if isinstance(bins,int):
                N=bins
            else:
                N = bins[d]
            step = (max-min) / N
            sample_bins[:,d] = (samples[:,d]-min) // step
            # make bin = N to bin = N-1
            idx = np.where(sample_bins[:,d] == N)
            sample_bins[:,d][idx] = N-1
        return sample_bins

    def shift(self):
        #calcualte initial similarity 
        target_hist, new_ranges = self.adaptive_range(self.data,self.target)
        init_similarity = hist_similarity(target_hist,self.data.hist)
        print(init_similarity)
        
        center = self.data.center
        next_center = self.next_center()
        x = np.arange(-1.6,1.6,0.16)
        y = np.arange(-1.6,1.6,0.16)
        z = np.arange(-1.6,1.6,0.16)
        index = []
        for xx in x:
            for yy in y:
                for zz in z:
                    index.append(np.array([xx,yy,zz])+center)
        similarity = []
        i = 0
        for next_center in index:
            #calculate new similarity
            self.data.set_center(next_center)
            self.data.update_hist()
            target_hist, new_ranges = self.adaptive_range(self.data,self.target)
            new_similarity = hist_similarity(target_hist,self.data.hist)
            similarity.append(new_similarity)
            i+=1
            print(i)
        print(len(similarity))
        vtk_similarity = numpy_support.numpy_to_vtk(np.array(similarity),False,VTK_FLOAT)
        img_vtk = vtkStructuredPoints()
        img_vtk.SetDimensions((20,20,20))
        img_vtk.GetPointData().SetScalars(vtk_similarity)

        writer = vtkXMLImageDataWriter()
        writer.SetFileName("test.vti")
        writer.SetInputData(img_vtk)
        writer.Write()

def hist_similarity(h1,h2):
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
    return np.sum(np.sqrt(h1*h2))

def nn_distance(pc1,pc2):
    pc1 = pc1[:,None,:]
    pc2 = pc2[None,:,:]
    d = (pc1 - pc2)**2
    d = np.sum(d,axis=-1)

    d1 = np.min(d,axis=-1)
    d2 = np.min(d,axis=-2)
    dis = np.concatenate((d1,d2))
    dis = np.mean(dis)
    return dis

def track_run(path,start,end,step,init_center,h,bins,model,device,dim,latent=True):
    center = init_center
    center_list = []
    center_list.append(center)
    # data = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(start)))
    # data = data[:,:dim]
    # start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
    # start_df = data_frame(data,3,center,h,bins,None)
    # m = start_df.near_pc.copy()
    # pc1 = m.copy()
    # pc1 = mean_sub(pc1)
    # print(center)
    # scatter_3d(pc1,center=center)
    for i in range(start,end+step-1,step):
        data = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(i)))
        data = data[:,:dim]
        # scatter_3d(data,50,350,50,center,False)
        # scatter_3d(data,50,350,50,center,True,"{:03d}.png".format(i))

        data_next = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(i+step)))
        data_next = data_next[:,:dim]

        if latent:
            start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
            target = latent_df(data_next,3,center,h,bins,None,model,device,dim)
        else:
            start_df = data_frame(data,3,center,h,bins,None)
            target = data_frame(data_next,3,center,h,bins,None)
        
        m = start_df.near_pc.copy()
        pc1 = m.copy()
        pc1 = mean_sub(pc1)
        # scatter_3d(pc1)

        pc2 = target.near_pc.copy()
        pc2 = mean_sub(pc2)
        # scatter_3d(pc2)

        # print(m.shape)
        print(center)

        ms = mean_shift(m,target,ite=20,dis=0.01)
        ms.shift()
        pc3 = target.near_pc.copy()
        pc3 = mean_sub(pc3)
        # scatter_3d(pc3)

        dis1 = nn_distance(pc1,pc2)
        dis2 = nn_distance(pc1,pc3)
        # if dis2< dis1:
        center = target.center
        center_list.append(center)

        print("original distance:",dis1)
        print("after meanshift:",dis2)
    # print(center_list)

