# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import quaternionic as qua
path_SBSD = "E:/github/SBSD_public_review/"
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)   
from core import sh_utils as shu
from dipy.core.gradients import  unique_bvals_tolerance
import json
import nibabel as nib
from scipy.special import eval_legendre

#This grid reconstructs estimated per-voxel per-direction impulses from estimated SH expansions.
# It does it once on an arbitrary sampling of the sphere where only z changes
# It does it once on the original sampling of the sphere which was used for the acquisition scheme.

#%% Paths  and parameters
base_hcp_path = os.path.join(*['E:\\',
              'dataHCP',
              ])
cache_path = os.path.join(path_SBSD, "CachedRotationDictionary")
L = 8
patient_id = "1007"
run_id= "MSMT_review_05" 
bval = 3000
base_patient_path = os.path.join(base_hcp_path, *['MGH_{0}_all'.format(patient_id),
'mgh_{0}'.format(patient_id)])
run_path = os.path.join(base_patient_path, *['diff','preproc','mri','SHEST_'+run_id, ]) 
save_path = os.path.join(run_path, 'recon_impulses2')
#%% Load data
print('Loading diffusion data...')

nifti_path = os.path.join(base_patient_path, *['diff','preproc','mri','diff_preproc.nii.gz'])

# Load nifti
full_img = nib.load(nifti_path)

# Get data
img = full_img.get_fdata(dtype = np.float32)
print('Loading diffusion data... Done!')


#%% Load FA map
print('Loading FA data...')

nifti_path = os.path.join(base_patient_path, *['diff','preproc','mri','DTI', 'FA.nii.gz'])

# Load nifti
full_fa = nib.load(nifti_path)

# Get data
fa = full_fa.get_fdata(dtype = np.float32)
print('Loading FA data... Done!')



#%% Load MNI mask (dwi)
mask_path = os.path.join(base_hcp_path, *['brain_masks','MNI',
                                       f'MNI_mask_{patient_id}.nii.gz'])
mask = nib.load(mask_path)
mask = mask.get_fdata(dtype = np.float32).astype('bool')



#%% Get the bvals and bvecs

bvals_path =os.path.join(base_hcp_path, *[f'MGH_{patient_id}_all',f'mgh_{patient_id}',
              'diff','preproc', 'bvals.txt'])
bvecs_path =os.path.join(base_hcp_path, *[f'MGH_{patient_id}_all',f'mgh_{patient_id}',
              'diff','preproc', 'bvecs_moco_norm.txt'])

bvals = np.loadtxt(bvals_path)
ubvals = unique_bvals_tolerance(bvals, tol = 70)
bval_idx = np.argmin(np.abs(ubvals-bval))
bvecs = np.loadtxt(bvecs_path )
mask_noB0 = bvals>100
xyz = bvecs[mask_noB0,:]

#Compute masks for isolating shells
mask_shells = np.zeros((ubvals.shape[0], bvals.shape[0]), dtype = 'bool')
for k,bvalue in enumerate(ubvals):
    mask_shells[k,:] = np.abs(bvals-bvalue)<70
z_basegrid = bvecs[:,2][mask_shells[bval_idx,:]]
idx_sort_z_basegrid = np.argsort(z_basegrid)
#%%
lin_data = img[mask,...][...,mask_shells[bval_idx,:]]
lin_data = lin_data/np.mean(img[mask,...][...,~mask_noB0], axis = -1, keepdims = True)
#%% Computation of spherical coordinates, quaternions of the grid ;
theta,phi,r = shu.Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
grid = np.zeros((theta.shape[0],2))
grid[:,0] = theta[:]
grid[:,1] = phi[:]
Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
xyz = shu.Sph2Cart(theta,phi,1)
    

#%% Load SH extimation
base_patient_path = os.path.join(base_hcp_path, *['MGH_{0}_all'.format(patient_id),
'mgh_{0}'.format(patient_id)])

#Real
weights_real_path = os.path.join(run_path, run_id+'_SHEST_weights_real.nii.gz')
weights_real_nifti = nib.load(weights_real_path)
weights_real = weights_real_nifti.get_fdata(dtype = np.float32)

num_peak = weights_real.shape[3]

#Load the metadata
with open(os.path.join(base_patient_path, *['diff','preproc','mri', 'SHEST_'+run_id, 'run_' + run_id + '.json'])) as f:
    metadata = json.load(f)

#%% Reshape the data and only keep voxels within the brain mask
lin_weights_real = weights_real[mask,...]

#%% Prepare a grid on which the impulses will be reconstructed
M = 200
z = np.linspace(-1,1,200)

# Compute the Yn0
Pn = np.zeros((M,L//2+1), dtype = 'float64')
for k,n in enumerate(range(0,L+1,2)):
    Pn[:,k] =  np.sqrt((2*n+1)/(4*np.pi)) * eval_legendre(n,z)
    
# Compute the Yn0 - basegrid
Pn_basegrid = np.zeros((z_basegrid.shape[0],L//2+1), dtype = 'float64')
for k,n in enumerate(range(0,L+1,2)):
    Pn_basegrid[:,k] =  np.sqrt((2*n+1)/(4*np.pi)) * eval_legendre(n,z_basegrid)
    

#%% Reconstruct the impulses
impulses = (lin_weights_real @ Pn.T).real

ws = 10
impulses = shu.pp_impulses(impulses, z, ws = ws, start_z = 0.9)
mask_corr = np.any(impulses<0, axis = -1)
impulses[mask_corr,:] = impulses[mask_corr,:] - np.min(impulses[mask_corr,:], axis = -1, keepdims = True)

#%% Sanity checks
sample = np.random.randint(impulses.shape[0])
fig,axs = plt.subplots(1,2)
fig.suptitle('Impulses Sample : ' + str(sample))
axs[0].plot(impulses[sample,0,:])
axs[1].plot(impulses[sample,1,:])
plt.show()

#%% 
store_impulses = np.zeros((*weights_real.shape[:3],impulses.shape[-2], impulses.shape[-1]))
store_impulses[mask,...] = impulses[...]


#%% Plot impulses at selected voxels
#(65,79,48) #In CC
#(50,79,68)
vox_x, vox_y, vox_z = (60,79,60)#  (50,79,68)
#vox_x, vox_y, vox_z = np.random.randint(30,70, size = 3)

def flip_coronal(tobeplot):
    tobeplot = np.swapaxes(tobeplot, 0,1).squeeze()
    tobeplot = np.flip(tobeplot, axis = 0)
    return tobeplot

#%% Check voxels position
slc_type = "frontal"

if(slc_type == 'frontal'):
    slc = [slice(None),slice(vox_y,vox_y+1),slice(None)]  
    swap_type = "frontal"
    
check_pos = np.zeros(fa.shape, dtype = 'bool')
check_pos[vox_x, vox_y, vox_z] = 1


plt.figure()
plt.imshow(flip_coronal(fa[tuple(slc)].squeeze()), origin = 'upper', cmap = 'binary')
plt.imshow(flip_coronal(check_pos[tuple(slc)].squeeze()), origin = 'upper', cmap = 'spring', 
            alpha = flip_coronal(check_pos[tuple(slc)].squeeze().astype('float32')))
plt.show()

#%% Plot the responses
colors = ['c', 'm', 'y']
imps = store_impulses[vox_x,vox_y,vox_z,...]
ylim = np.max(imps)*1.05
fig,axs = plt.subplots(1,imps.shape[0])
for k in range(imps.shape[0]):  
    axs[k].plot(z, imps[k,:], color = colors[k])
    axs[k].set_ylim(-0.01,ylim)
plt.show()


#%% Store the reconstructed impulses
if(not(os.path.exists(save_path))):
    os.makedirs(save_path)
header = full_img.header
header.dim = store_impulses.shape
nifti = nib.Nifti1Image(store_impulses, full_img.affine, header=header)
nib.save(nifti, os.path.join(save_path, '{0}_recon_impulses.nii.gz').format(run_id))



#%% ======================================================================================== BASE GRID


#%% Reconstruct the impulses
impulses_bg = (lin_weights_real @ Pn_basegrid.T).real

ws = 10

impulses = shu.pp_impulses(impulses_bg[:,:,idx_sort_z_basegrid], z_basegrid[idx_sort_z_basegrid] , ws = ws, start_z = 0.9)
invert_permut = np.argsort(idx_sort_z_basegrid)
impulses = impulses[:,:,invert_permut]


mask_corr = np.any(impulses<0, axis = -1)
impulses[mask_corr,:] = impulses[mask_corr,:] - np.min(impulses[mask_corr,:], axis = -1, keepdims = True)

#%% Sanity checks
sample = np.random.randint(impulses.shape[0])
fig,axs = plt.subplots(1,2)
fig.suptitle('Impulses Sample : ' + str(sample))
axs[0].plot(impulses[sample,0,:])
axs[1].plot(impulses[sample,1,:])
plt.show()

#%% 
store_impulses = np.zeros((*weights_real.shape[:3],impulses.shape[-2], impulses.shape[-1]))
store_impulses[mask,...] = impulses[...]


#%% Plot impulses at selected voxels
#(65,79,48) #In CC
#(50,79,68)
vox_x, vox_y, vox_z = (45,79,60)#  (50,79,68)


def flip_coronal(tobeplot):
    tobeplot = np.swapaxes(tobeplot, 0,1).squeeze()
    tobeplot = np.flip(tobeplot, axis = 0)
    return tobeplot

#%% Check voxels position
slc_type = "frontal"

if(slc_type == 'frontal'):
    slc = [slice(None),slice(vox_y,vox_y+1),slice(None)]  
    swap_type = "frontal"
    
check_pos = np.zeros(fa.shape, dtype = 'bool')
check_pos[vox_x, vox_y, vox_z] = 1


plt.figure()
plt.imshow(flip_coronal(fa[tuple(slc)].squeeze()), origin = 'upper', cmap = 'binary')
plt.imshow(flip_coronal(check_pos[tuple(slc)].squeeze()), origin = 'upper', cmap = 'spring', 
            alpha = flip_coronal(check_pos[tuple(slc)].squeeze().astype('float32')))
plt.show()

#%% Plot the responses
colors = ['c', 'm', 'y']
imps = store_impulses[vox_x,vox_y,vox_z,...]
ylim = np.max(imps)*1.05
fig,axs = plt.subplots(1,imps.shape[0])
for k in range(imps.shape[0]):  
    axs[k].plot(z_basegrid[idx_sort_z_basegrid], imps[k,idx_sort_z_basegrid], color = colors[k])
    axs[k].set_ylim(-0.01,ylim)
plt.show()


#%% Store the reconstructed impulses
if(not(os.path.exists(save_path))):
    os.makedirs(save_path)
header = full_img.header
header.dim = store_impulses.shape
nifti = nib.Nifti1Image(store_impulses, full_img.affine, header=header)
nib.save(nifti, os.path.join(save_path, '{0}_recon_basegrid_impulses.nii.gz').format(run_id))




