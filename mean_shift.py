import os
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.ckdtree import cKDTree
from sklearn.decomposition import PCA
from model import AE

from process_data import *

class LatentRetriever():
    def __init__(self,data,model,args,latent_dimension,pca):
        source = args.source
        if source == "fpm":
            mean = [0, 0, 5, 23.9, 0, 0, 0.034]
            std = [2.68, 2.68, 3.09, 55.08, 0.3246, 0.3233, 0.6973]
        elif source == "cos":
            mean = [30.4, 32.8, 32.58, 0, 0, 0, 0, 0, 0, -732720]
            std = [18.767, 16.76, 17.62, 197.9, 247.2, 193.54, 420.92, 429, 422.3, 888474]
        data = normalize(data,mean,std)
        coord = data[:,:3]
        self.kd = cKDTree(coord,leafsize=30)
        self.mask = np.full((len(data),),False)
        self.args = args
        self.model = model
        self.data = data
        self.latent = np.zeros((len(data),latent_dimension))
        self.pca = pca
    def retriev(self,idx):
        args = self.args
        data = self.data
        kd = self.kd
        model = self.model
        not_infered = np.logical_not(self.mask)
        inference_idx = np.logical_and(not_infered,idx)
        #set infered
        self.mask = np.logical_or(self.mask,idx)
        if np.sum(inference_idx)>0:
            center = data[inference_idx,:3]
            _, nn = kd.query(center,args.k,n_jobs=-1)
            batch = data[nn][:,:,:args.dim]
            batch[:,:,:3] -= center[:,None,:]
            # batch -= data[inference_idx][:,None,:]
            batch =torch.FloatTensor(batch).cuda()
            model.eval()
            with torch.no_grad():
                latent = model.encode(batch).cpu().detach()
            self.latent[inference_idx] = self.pca.transform(latent)
        return self.latent[idx]

def filter(data,c1,c2,multiple = 1):
    # enlarge the area by multiple
    c1 = np.array(c1)
    c2 = np.array(c2)
    c0 = (c1+c2)/2
    c1 = c0 + multiple * (c1-c0)
    c2 = c0 + multiple * (c2-c0)
    x1,y1,z1 = c1
    x2,y2,z2 = c2
    condx = np.logical_and(data[:,0]>x1,data[:,0]<x2)
    condy = np.logical_and(data[:,1]>y1,data[:,1]<y2)
    condz = np.logical_and(data[:,2]>z1,data[:,2]<z2)
    cond = np.logical_and(condx,condy)
    cond = np.logical_and(cond,condz)
    return cond

def mean_shift_track(
    t1,t2,c1,c2, latent=False, 
    h=1,bins=10,eps=1e-3,ite=300, 
    model=None,args=None,latent_dimension=None,pca=None):

    # crop an approximate area
    t1 = t1[filter(t1,c1,c2,3)]
    d1_idx = filter(t1,c1,c2)
    d1 = t1[d1_idx]
    #show initial 
    x1,y1,z1 = c1
    x2,y2,z2 = c2
    center = ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)
    scatter_3d(t1,threshold=10,center=center)

    t2 = t2[filter(t2,c1,c2,3)]
    d2_idx = filter(t2,c1,c2)
    d2 = t2[d2_idx]
    center = np.mean(d1[:,:3],axis=0)
    if not latent:
        d1_attr = d1[:,3:]
    else:
        d1_lr = LatentRetriever(t1,model,args,latent_dimension,pca)
        d1_attr = d1_lr.retriev(d1_idx)
    w = 1 - np.sum(((d1[:,:3]-center)/h)**2,axis=-1)
    w[w<0] = 1e-5
    hist1,boundary = np.histogramdd(
        d1_attr,
        bins=bins,
        density=True,
        weights=w,
    )
    ranges = []
    for b in boundary:
        ranges.append((b[0],b[-1]))
    if not latent:
        d2_attr = d2[:,3:]
    else:
        d2_lr = LatentRetriever(t2,model,args,latent_dimension,pca)
        d2_attr = d2_lr.retriev(filter(t2,c1,c2))
    w = 1 - np.sum(((d2[:,:3]-center)/h)**2,axis=-1)
    w[w<0] = 1e-5
    hist2,_ = np.histogramdd(
        d2_attr,
        range=ranges,
        bins=bins,
        density=True,
        weights=w,
    )
    current_ite = 0
    reach_eps = False
    while(True):
        #calcualte initial similarity 
        init_similarity = hist_similarity(hist1,hist2)
        weights = np.sqrt(hist1/(hist2+1e-10))
        near_bins = []
        for i in range(d2_attr.shape[1]):
            bin_number = np.digitize(d2_attr[:,i],boundary[i],right=False)
            bin_number -= 1
            bin_number[bin_number==bins] = bins-1
            near_bins.append(bin_number)
        near_bins = tuple(near_bins)
        new_center = np.average(d2[:,:3],axis=0,weights=weights[near_bins])
        
        while (True):
            shift_vector = new_center - center
            # if the shift length is smaller than eps directly end 
            if np.sum(shift_vector **2) < eps ** 2:
                reach_eps = True
                break
            # update selection
            c1 += shift_vector
            c2 += shift_vector
            d2_idx = filter(t2,c1,c2)
            d2 = t2[d2_idx]
            if not latent:
                d2_attr = d2[:,3:]
            else:
                d2_attr = d2_lr.retriev(d2_idx)
            w = 1 - np.sum(((d2[:,:3]-new_center)/h)**2,axis=-1)
            w[w<0] = 1e-5
            hist2,_ = np.histogramdd(
                d2_attr,
                range=ranges,
                bins=bins,
                density=True,
                weights=w,
            )
            new_similarity = hist_similarity(hist1,hist2)
            # fine tune shift length
            if (new_similarity > init_similarity):
                break
            else:
                new_center = (center + new_center)/2
        center = new_center
        #check for ending conditions
        if reach_eps:
            break
        current_ite+=1
        if current_ite == ite:
            break
    # show final
    x1,y1,z1 = c1
    x2,y2,z2 = c2
    center = ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)
    scatter_3d(t2,threshold=10,center=center)

    return c1,c2
    
def hist_similarity(h1,h2):
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
    return np.sum(np.sqrt(h1*h2))

def weighted_hist(near_coord,near_attr, center, h,bins,ranges):
    weights = 1 - np.sum(((near_coord-center)/h)**2,axis=-1)
    hist = np.histogramdd(
        near_attr,
        bins = bins,
        range = ranges,
        weights = weights,
        density = True)
    print(hist[0])
    hist[0][hist[0]<0] = 0
    return hist[0]

MAX = 360

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
        self.latent_length = 2

        latent = np.zeros((len(data),self.latent_length))
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
                x.append(self.data[tt][:,:-self.latent_length])
            x = prepare_for_model(x,device,3,dim)
            # print(x[0].shape)
            with torch.no_grad():
                y = model.encode(x)
            latent = y.detach().cpu().numpy()   
            # print(latent.shape) 
            self.data[not_cal,-self.latent_length:] = latent[:,:self.latent_length]
            self.near_coord = self.coord[self.near]
            self.near_attr = self.attr[self.near]
            self.near_pc = self.data[self.near]

def get_latent(model,x:np.ndarray,device,dim):
    x = prepare_for_model([x],device,3,dim)
    with torch.no_grad():
        y = model.encode(x)
    return y.detach().cpu().numpy()        

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
        i = 0
        while(True):
            #calcualte initial similarity 
            target_hist, new_ranges = self.adaptive_range(self.data,self.target)
            init_similarity = hist_similarity(target_hist,self.data.hist)
            # print(init_similarity)
            
            center = self.data.center
            next_center = self.next_center()

            count = 0 
            while (True):
                #calculate new similarity
                self.data.set_center(next_center)
                self.data.update_hist()
                target_hist, new_ranges = self.adaptive_range(self.data,self.target)
                new_similarity = hist_similarity(target_hist,self.data.hist)
                count += 1
                if (new_similarity > init_similarity or count == 13):
                    break
                else:
                    next_center = (center + next_center)/2

            # t1 = datetime.now()
            # self.data.set_center((next_center+self.data.center)/2)
            # self.data.update_hist()

            i+=1
            if i == self.ite or np.sqrt(np.sum((center-next_center)**2))<self.dis:
                break
        print("Mean_shift_next_center",self.data.center)
        return self.data

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
        scatter_3d(data,50,350,50,center,False)
        # scatter_3d(data,50,350,50,center,True,"{:03d}.png".format(i))

        data_next = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(i+step)))
        data_next = data_next[:,:dim]

        t0 = datetime.now()
        if latent:
            start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
            target = latent_df(data_next,3,center,h,bins,None,model,device,dim)
        else:
            start_df = data_frame(data,3,center,h,bins,None)
            target = data_frame(data_next,3,center,h,bins,None)
        
        m = start_df.near_pc.copy()
        if i == start:
            pc1 = m.copy()
            pc1 = mean_sub(pc1)
            pc1= pc1[:,:4]
            # print(pc1)
        # pc1 = m.copy()
        # pc1 = mean_sub(pc1)
        # scatter_3d(pc1)

        pc2 = target.near_pc.copy()
        pc2 = mean_sub(pc2)
        pc2 = pc2[:,:4]
        # print(pc2.shape)
        # scatter_3d(pc2)

        ms = mean_shift(m,target,ite=20,dis=0.01)
        ms.shift()
        t1 = datetime.now()
        print("total time: ",t1-t0)
        # pc3 = target.near_pc.copy()
        # pc3 = mean_sub(pc3)
        # scatter_3d(pc3)

        dis1 = nn_distance(pc1,pc2)
        # dis2 = nn_distance(pc1,pc3)
        # if dis2< dis1:
        center = target.center
        center_list.append(dis1)

        print("original distance:",dis1)
        # print("after meanshift:",dis2)
    print(center_list)


    

if __name__ == "__main__":
    
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'
        
    state_dict = torch.load("states/CP1.pth")
    state = state_dict['state']
    args = state_dict['config']
    print(args)
    # halo_info = halo_reader(data_path+"/ds14_scivis_0128/rockstar/out_47.list")
    model = AE(args).float().cuda()
    model.load_state_dict(state)

    # c1 = (3.2,0.25,5.5)
    # c2 = (4.3,1.25,6.5)
    # c1 = (-2.5,-2.5,5.2)
    # c2 = (-1.5,-1.5,6.2)
    # c1 = (0,3.5,6.5)
    # c2 = (1,4.5,7.5)
    # c1 = (-1,-1,7.9)
    # c2 = (-0.25,-0.25,8.5)
    c1 = (0,-2,2.5)
    c2 = (2,0,3.5)
    with open("pca","rb") as f:
        pca = pickle.load(f)
    print(c1,c2)

    for i in range(35,45):
        data_directory = data_path+"/2016_scivis_fpm/0.44/run41/0{}.vtu".format(i)
        data1 = vtk_reader(data_directory)
        data1 = data1[:,:7]
        scatter_3d(data1[::3],vmin=10,threshold=10)
        data_directory = data_path+"/2016_scivis_fpm/0.44/run41/0{}.vtu".format(i+1)
        data2 = vtk_reader(data_directory)
        data2 = data2[:,:7]
        c1,c2 = mean_shift_track(data1,data2,c1,c2,True,h=1,bins=3,model=model,args=args,latent_dimension=5,pca=pca)
        print(c1,c2)
