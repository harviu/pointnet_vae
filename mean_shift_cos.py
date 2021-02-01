import os
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.ckdtree import cKDTree
from sklearn.decomposition import PCA
from model import AE
from sklearn.cluster import KMeans
from thingking import loadtxt

from process_data import *
from simple import show

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
            batch =torch.FloatTensor(batch).cuda()
            model.eval()
            with torch.no_grad():
                latent = model.encode(batch)
                if args.have_label:  #change for new layer
                    latent = model.cls[:6](latent)
                latent = latent.cpu().detach()
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
    data1,data2,c1,c2, latent=False, 
    h=1,bins=10,eps=1e-4,ite=30, 
    model=None,args=None,latent_dimension=None,pca=None):

    # crop an approximate area
    t1 = data1[filter(data1,c1,c2,5)]
    d1_idx = filter(t1,c1,c2)
    d1 = t1[d1_idx]

    t2 = data2[filter(data2,c1,c2,5)]
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

    return c1,c2, current_ite
    
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

def get_benchmark(path, start,end,index):
    center_list = []
    r = []
    for i in range(start,end+1):
        ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
            loadtxt(path+"/ds14_scivis_0128/rockstar/out_{:02d}.list".format(i-2), unpack=True)
        order = list(ID).index(index) # get order of the halo idx
        this_center = (x[order],y[order],z[order]) # get this center
        this_r = Rvir[order] # get this radius
        center_list.append(this_center)
        r.append(this_r/1000)
        index = DescID[order] # get next index
        if index == -1:
            print("halo disappear")
            break
    # print(center_list)
    return center_list, r

if __name__ == "__main__":
    # np.seterr(all='raise')
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    directory = "states_saved/cos_k64_v256/"
    # directory = "states_saved/cos_k128_v512/"
    state_dict = torch.load(directory + "/CP35.pth")
    state = state_dict['state']
    args = state_dict['config']
    print(args)
    halo_info = halo_reader(data_path+"/ds14_scivis_0128/rockstar/out_47.list")
    model = AE(args).float().cuda()
    model.load_state_dict(state)
    data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
    data = sdf_reader(data_directory)

    pca = pickle.load(open( directory+"/pca_late", "rb" ))
    all_distance = []
    c_list = []
    for j in range(3633,3634):
        start = 49
        track_length = 10
        center,r = get_benchmark(data_path,start,start+track_length,j)
        print("halo_index:",j)
        # print("center",center)
        track_length = len(r) -1
        margin = 0.05
        xyz = center[0]
        width = r[0] + margin
        c1 = (xyz[0]-width,xyz[1]-width,xyz[2]-width)
        c2 = (xyz[0]+width,xyz[1]+width,xyz[2]+width)
        # time_list = []
        distance_list = np.zeros((track_length))
        time_step = 0
        breaker = True

        for i in range(start,start+track_length):
            data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.{}00'.format(i)
            data1 = sdf_reader(data_directory)
            data1 = data1[:,:args.dim]

            data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.{}00'.format(i+1)
            data2 = sdf_reader(data_directory)
            data2 = data2[:,:args.dim]

            # try:
            c1,c2,iteration_number = mean_shift_track(data1,data2,c1,c2,True,h=1,bins=2,model=model,args=args,latent_dimension=4,pca=pca)
            # except:
            #     breaker = False
            #     break
            c = (np.array(c1)+np.array(c2))/2
            c_list.append(c)
            distance = np.sqrt(np.sum((np.array(center[time_step+1]) - c) ** 2))
            distance_list[time_step] += distance
            time_step+=1
        # if not breaker:
        #     continue
        all_distance.append(distance_list)
        distance_list /= np.array(r[1:])
        print(distance_list)
        # print(c_list)
    array_dict = {
        "index": np.arange(track_length)
    }
    print(np.array(center[1:]).shape)
    print(np.array(c_list).shape)
    gt_vtk = numpy_to_vtk(np.array(center[1:]),array_dict)
    track_vtk = numpy_to_vtk(np.array(c_list),array_dict)
    vtk_write(gt_vtk,"gt.vtu")
    vtk_write(track_vtk,"track.vtu")


