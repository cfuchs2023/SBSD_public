# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pickle
import spherical as sph
import quaternionic as qua
import json


path_SBSD = "E:/github/SBSD_public/" #Put the path where you git cloned SBSD_public
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)   
 

from core import sh_utils as shu
from core import DataGenerator as DG
from core import SBSD as sbsd


#%% Loading the DWI data
base_data_path = os.path.join(path_SBSD,"data")
scheme_name = "01scheme_200"
task_name = scheme_name + '_SNR100'
data_run_id = 0

nsamples = 100
Data = DG.DataLoader(base_data_path, 
             task_name, 
             data_run_id,
             num_samples = nsamples,
                 num_fasc = 2)

bval = 5000
rep_id = "Small_00"
FORCE_OVERWRITE = True

#%% Loading the estimation of orientation
dir_estimation_type = 'GROUNDTRUTH' #SBSD, CSD or GROUNDTRUTH

if(dir_estimation_type == 'SBSD'):
    run_id = 'rep_' + rep_id + '_' + task_name + "_" +"_bval_" +str(bval)
    path_estimation = os.path.join(path_SBSD, 'Reports', 'RotationEstimationMC_SBSD')
    with open(os.path.join(path_estimation, run_id + '.pickle'), 'rb') as f:
        dir_est_data = pickle.load(f)
    with open(os.path.join(path_estimation, run_id + '.json')) as f:
        dir_est_metadata = json.load(f)
        idx_degree_used_for_u_est = 2
    dir_est = dir_est_data['estimated_us'][:,idx_degree_used_for_u_est,:,:]
elif(dir_estimation_type == 'GROUNDTRUTH'):
    dir_est = Data.data['orientations']
     

#%%
base_dic_path = "E:/github/SBSD_private/dictionaries/TestSchemes"
dic_name = "TestSchemesDic1.pkl"
dictionary_path = os.path.join(base_dic_path, dic_name)
with open(dictionary_path, 'rb') as f:
    full_dic = pickle.load(f)
dic = full_dic[scheme_name]


#%% Computation of spherical coordinates, quaternions of the grid 
bvals = dic['bvals']
ubvals = np.unique(bvals)
bvecs = dic['sch_mat'][:,0:3]
mask_noB0 = bvals>100

#Choose the bval
bval = ubvals[3]
idx_bval = np.argmin(np.abs(bval-ubvals))-1
mask_bval = np.abs(bvals[mask_noB0] - bval)<100
xyz = bvecs[mask_noB0,:]

theta,phi,r = shu.Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
grid = np.zeros((theta.shape[0],2))
grid[:,0] = theta[:]
grid[:,1] = phi[:]
Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
xyz = shu.Sph2Cart(theta,phi,1)


#%% Parameters
if(dir_estimation_type == 'SBSD'):
    L = dir_est_metadata['L'] #Degree for the computation of the expansion of the signal
    c_smooth = dir_est_metadata['c_smooth'] #Smooth factor for the computation of the projection matrix over SH
    t_smooth = 1e-3 #For tikhonov regularization in LS solving in OMP
else:
    L = 8
    c_smooth = 1e-9
    t_smooth = 1e-9
#Compute basis matrix and projection matrix
wig = sph.Wigner(L)
Ymn = wig.sYlm(0, Rs_grid[mask_bval,:]) 
n_sh = np.array([n for n in range(L+1) for m in range(-n,n+1)])
proj_mat, c_proj_mat = shu.get_SH_proj_mat(L,None, n_sh, smooth = None, 
                                           cSH = Ymn, c_smooth = c_smooth)

#%% Normalize data with values at b = 0
x0 = np.max(Data.data['DWI_noisy'][:,~mask_noB0,], axis = 1, keepdims = True)
x = Data.data['DWI_noisy'][:,mask_noB0][:,mask_bval]/x0
x_noiseless = Data.data['DWI'][:,mask_noB0][:,mask_bval]/x0

#%% COmpute SH expansion of the data
cSH_x = (c_proj_mat @ x.T).T
recon_x = (Ymn @ cSH_x.T).T

#%% Compute the SH expansion coefficients estimation
#dir_est_data['estimated_us'] has shape (nsamples, len(Ns), n_peaks, 3)
#the direction estimation was performed with multiple degrees and so we need to choose one
#you can look in dir_est_metadata['results'] to help you choose
weights,Ru = sbsd.ComputeCanonicalSignals(L, cSH_x, 2,dir_est,
                                        t_smooth, verbosity = 1)



#%% Load ground truth expansion coefficients
gt_SH = full_dic['SH_expansion']
gt_L = int(np.sqrt(gt_SH.shape[2] - 1))
SH_canon_mask = np.array([m == 0 and n%2 == 0 for n in range(0,gt_L+1) for m in range(-n,n+1)])
gt_SH_coeffs = gt_SH[:,:,SH_canon_mask]


#%% 
gt_weights = gt_SH_coeffs[Data.data['IDs'],:,:][:,:,idx_bval,0:weights.shape[-1]] #C_{0,k}^{n}
gt_weights = gt_weights * Data.data['nus'][:,:,np.newaxis] #\nu_k C_{0,k}^{n}
err = np.abs(np.abs(gt_weights) - np.abs(weights)) #Investigate tgis 
rel_errs = err/np.abs(gt_weights)
mean_err = np.mean(err, axis = 0)
mean_rel_err = np.mean(rel_errs, axis = 0)


data_results = {}
data_results['mean_rel_errs'] = mean_rel_err.tolist()
data_results['mean_errs'] = mean_err.tolist()


#%%
data_results['mean_errs'] = np.mean(err, axis = 0).tolist()
print(data_results)


#%% SAve results
save_path = "E:\\github\\SBSD_private\\Reports\\SHEstimation_SBSD"
if(not(os.path.exists(save_path))):
    os.makedirs(save_path)
    
    
metadata = {
    'task_name':task_name,
    'dic_path':dictionary_path,
    'scheme_name':scheme_name,
    'base_data_path':scheme_name,
    'data_run_id':data_run_id,
    'L':L,
    'bval':bval,
    'c_smooth':c_smooth,
    't_smooth':t_smooth,
    'dir_estimation_type':dir_estimation_type
    
    }

if(not(FORCE_OVERWRITE) and os.path.exists(os.path.join(save_path, run_id+'.json'))):
    raise RuntimeError('File already exists.')
 
run_id = 'rep_' + rep_id + '_' + task_name + "_" +"_bval_" + str(math.trunc(bval*1e-6))
with open(os.path.join(save_path, run_id+'.json'), 'w') as f:
    json.dump(metadata, f, indent = 2)

data = {'gt_weights':gt_weights.tolist(),
        'est_weights':weights.tolist(),
        'dir_est':dir_est.tolist(),
        'dir_gt':Data.data['orientations'].tolist()}

with open(os.path.join(save_path, run_id+'.pickle'), 'wb') as f:
    pickle.dump(data,f)