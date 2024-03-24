# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
path_SBSD = "E:/github/SBSD_public_review/"
from dipy.core.gradients import  unique_bvals_tolerance

#%% Paths  and parameters
base_hcp_path = os.path.join(*['E:\\',
              'dataHCP',
              ])
linux_base_hcp_path = "/e/dataHCP/"
cache_path = os.path.join(path_SBSD, "CachedRotationDictionary")
L = 8
patient_id = "1010"
run_id= "MSMT_review_05" 
bval = 3000
base_patient_path = os.path.join(base_hcp_path, *['MGH_{0}_all'.format(patient_id),
'mgh_{0}'.format(patient_id)])
run_path = os.path.join(base_patient_path, *['diff','preproc','mri','SHEST_'+run_id, ]) 
imp_path = os.path.join(run_path, 'recon_impulses2')
load_dwi = True
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
    
mask_bval = mask_shells[bval_idx,:]
z_basegrid = bvecs[:,2][mask_shells[bval_idx,:]]
idx_sort_z_basegrid = np.argsort(z_basegrid)

#%% Store the relevant bvecs and bvals
new_bvecs = np.zeros((np.sum(mask_bval)+1, 3), dtype = 'float32')
new_bvecs[1:,:] = bvecs[mask_bval,:]
new_bvals = np.concatenate(([0], bvals[mask_bval]))

#%% Write the new_bvals an bvecs
new_path = os.path.join(imp_path, 'PD_DTI' )
if(not(os.path.exists(new_path))):
    os.makedirs(new_path)
    
with open(os.path.join(new_path, 'bvecs.txt'), 'w') as f:
    np.savetxt(f, new_bvecs)
    
with open(os.path.join(new_path, 'bvals.txt'), 'w') as f:
    np.savetxt(f, new_bvals)
   
    
#%% Load mask
if(patient_id != '1007' and patient_id != '1010'):
    mask_path = os.path.join(base_hcp_path, *['brain_masks', 'MNI', f'MNI_mask_{patient_id}.nii.gz'])
    mask_nifti = nib.load(mask_path)
    mask = mask_nifti.get_fdata(dtype = np.float32).astype('bool')
else:
    mask_path = os.path.join(base_hcp_path, *['brain_masks', 'MNI', f'MNI_mask_{patient_id}_edit.nii.gz'])
    mask_nifti = nib.load(mask_path)
    mask = mask_nifti.get_fdata(dtype = np.float32).astype('bool')

#%% Load dwi
if(load_dwi):
    nifti_path = os.path.join(base_patient_path, *['diff','preproc','mri','diff_preproc.nii.gz'])
    full_img = nib.load(nifti_path)
    img = full_img.get_fdata(dtype = np.float32)
    data_linear = img[mask,...]
    
    # Get good B0
    all_idx_vols = np.array([k for k in range(img.shape[-1])])
    idx_start_slice = all_idx_vols[mask_bval][0]
    idx_end_slice = all_idx_vols[mask_bval][-1]
    good_idx_b0 = all_idx_vols[idx_start_slice:idx_end_slice][~mask_noB0[idx_start_slice:idx_end_slice]]
    
    
    means_good_b0s = np.mean(data_linear[...,good_idx_b0], axis = -1)
    
#%% Store the signal for FA computation on one shell only
if(load_dwi):
    meansB0 = np.mean(data_linear[:,~mask_noB0], axis = -1)
    mask_B0means = np.abs(meansB0)>1e-6
    data_linear[mask_B0means,:] = data_linear[mask_B0means,:]/meansB0[mask_B0means][:,np.newaxis]
    new_img = np.ones((*img.shape[0:3], np.sum(mask_bval)+1), dtype = 'float32')
    new_img[mask,1:] = data_linear[:, mask_bval]
    
    header = full_img.header
    header.dim = new_img.shape
    nifti = nib.Nifti1Image(new_img, mask_nifti.affine, header=header)
    nib.save(nifti, f"{base_hcp_path}/MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/DTI_bval{bval}/single_shell_sig.nii.gz")

    
    
    

#%% Load the impulses reconstructed on the base grid
nifti = nib.load(os.path.join(imp_path, f'{run_id}_recon_basegrid_impulses.nii.gz'))
store_impulses = nifti.get_fdata(dtype = np.float32)
impulses = store_impulses[mask,...]
norm_impulses = np.linalg.norm(impulses, axis = -1)
num_fasc = store_impulses.shape[-2]

x_s, y_s, z_s, n_p, d_s = store_impulses.shape
#%% add the B0 of 1 at each impulse : everything was already normalized with B0 values
new_store_impulses = np.ones((x_s, y_s, z_s, n_p, d_s+1), dtype = 'float32')
new_impulses = np.ones((impulses.shape[0], impulses.shape[1], impulses.shape[2]+1), dtype = 'float32')

if(load_dwi):
    new_impulses[:,:,0] = means_good_b0s[:,np.newaxis]
new_impulses[:,:,1:] = impulses[...]
new_store_impulses[mask,...] = new_impulses

for k in range(impulses.shape[-2]):
    header = nifti.header
    header.dim = new_store_impulses[...,k,:].shape
    nifti = nib.Nifti1Image(new_store_impulses[...,k,:], nifti.affine, header=header)
    nib.save(nifti, os.path.join(new_path, '{0}_recon_basegrid_impulse_{1}.nii.gz').format(run_id,1+k))

#%% Utilities for generating mrtrix commands


def generate_dwi2tensor(patient_id, base_hcp_path, run_id, num_impulse = 1):
    base = f'''dwi2tensor  
    -fslgrad {base_hcp_path}MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/bvecs.txt 
            {base_hcp_path}MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/bvals.txt 
    -mask {base_hcp_path}brain_masks/MNI/MNI_mask_{patient_id}.nii.gz    
    {base_hcp_path}MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/{run_id}_recon_basegrid_impulse_{num_impulse}.nii.gz
    /e/dataHCP/MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/{run_id}_tensor_{num_impulse}.nii.gz
    '''
    return base.replace('\n', ' ')



def generate_tensor2fa(patient_id, base_hcp_path, run_id, num_impulse = 1, metric = 'fa'):
    base = f'''tensor2metric   
    -mask {base_hcp_path}brain_masks/MNI/MNI_mask_{patient_id}.nii.gz 
    -{metric} {base_hcp_path}MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/{run_id}_{metric}_{num_impulse}.nii.gz
    {base_hcp_path}MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/{run_id}_tensor_{num_impulse}.nii.gz
    '''
    return base.replace('\n', ' ')


#%% this cell generates the mrtrix commands needed to generate fa maps 
# you need to run those commands with MRtrix before running the next cell
for k in range(3):
    print(generate_dwi2tensor(patient_id, linux_base_hcp_path, run_id, num_impulse = k+1))
print('\n\n\n ================================== \n\n\n')


for metric in ['fa', 'rd', 'ad']:
    for k in range(3):
        print(generate_tensor2fa(patient_id, linux_base_hcp_path, run_id, num_impulse = k+1, metric = metric))


#%% Delinte API : fusing the maps in one .nii.gz to easily use UNRAVEL later on
for metric in ['fa', 'rd', 'ad']:
    num_fasc = 3
    
    all_metric = np.zeros((*mask_nifti.shape[0:3], num_fasc))
    for k in range(num_fasc):
        path = f"{base_hcp_path}/MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/{run_id}_{metric}_{k+1}.nii.gz"
        nifti = nib.load(path)
        all_metric[...,k] = nifti.get_fdata(dtype = 'float32')
        
    header = mask_nifti.header
    header.dim = all_metric.shape
    nifti = nib.Nifti1Image(all_metric, mask_nifti.affine, header=header)
    nib.save(nifti, f"{base_hcp_path}/MGH_{patient_id}_all/mgh_{patient_id}/diff/preproc/mri/SHEST_{run_id}/recon_impulses2/PD_DTI/{run_id}_{metric}_all.nii.gz")
