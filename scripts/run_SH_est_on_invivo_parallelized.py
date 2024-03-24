# -*- coding: utf-8 -*-
if(__name__ == '__main__'):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import spherical as sph
    import quaternionic as qua
    import sys
    import nibabel as nib
    from dipy.core.gradients import  unique_bvals_tolerance
    import json 
    import scipy
    path_SBSD = "E:/github/SBSD_public_review/" #Put the path where you git cloned SBSD_public
    if(not(path_SBSD in sys.path)):
        sys.path.insert(0,path_SBSD)   
    
    from core import sh_utils as shu
    from core import SBSD as sbsd
    
    from dask.diagnostics import ProgressBar
    import dask
    import time
    
    #%% Base paths
    base_hcp_path = os.path.join(*['E:\\',
                  'dataHCP',
                  ]) #Put the path where you stored data from the hcp
    patient_id = "1007"
    num_processes = 28
    dir_est_type = 'MSMTCSD'
    fix_number_of_peaks = False
    run_id = 'MSMT_review_05'
    store_residuals = True
    #%% Build paths
    base_patient_path = os.path.join(base_hcp_path, *['MGH_{0}_all'.format(patient_id),
    'mgh_{0}'.format(patient_id)])
    save_path = os.path.join(base_patient_path,*['diff','preproc','mri','SHEST_{0}'.format(run_id)])
    if(not(os.path.exists(save_path))):
        os.makedirs(save_path)
    diff_preproc_path = os.path.join(base_patient_path, *['diff','preproc','mri'],)
    nifti_path = os.path.join(diff_preproc_path, 'diff_preproc.nii.gz')
    #%% Parameters
    L = 8 #Degree for the computation of the expansion
    c_smooth = 1e-3 #For Laplace Beltrami regularization in the computation of the projection matrix
    t_smooth = 1e-4
    bval = 3000
    save_result_as_nifti = True
    
    # Slice on which the algorithm will be performed : None everyhwere -> perform on the complete volume (masked by a brain mask)
    slc = [slice(None),slice(None),slice(None)]  # Create a list of slices for each dimension
    #slc = [slice(None),slice(None),slice(20,40)] #Reduced volume for testing
    
    #Slice for vizualisation
    visu_slc = [slice(None),slice(80,81),  slice(None)]  # Create a list of slices for each dimension
    visu_slc_diff = visu_slc + [slice(0,1)] #Visualization of first diff volume

        
    
    #%% Load data
    print('Loading diffusion data...')
    
    # Load nifti
    full_img = nib.load(nifti_path)
    
    # Get data
    img = full_img.get_fdata(dtype = np.float32)
    print('Loading diffusion data... Done!')
    
    
    #%% Load mask (dwi)
    #Load MNI mask
    if(patient_id != '1007'):
        mask_path = os.path.join(base_hcp_path, *['brain_masks', 'MNI', f'MNI_mask_{patient_id}.nii.gz'])
        mask_nifti = nib.load(mask_path)
        mask = mask_nifti.get_fdata(dtype = np.float32).astype('bool')
    else:
        mask_path = os.path.join(base_hcp_path, *['brain_masks', 'MNI', f'MNI_mask_{patient_id}_edit.nii.gz'])
        mask_nifti = nib.load(mask_path)
        mask = mask_nifti.get_fdata(dtype = np.float32).astype('bool')
        
        

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
         
    

    #%% Computation of spherical coordinates, quaternions of the grid ;
    theta,phi,r = shu.Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
    grid = np.zeros((theta.shape[0],2))
    grid[:,0] = theta[:]
    grid[:,1] = phi[:]
    Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
    xyz = shu.Sph2Cart(theta,phi,1)
        
    #%% Extract data in mask into array of type (voxels, directions)
    slc_img = img[tuple(slc+[slice(None)])]
    slc_mask = mask[tuple(slc)]
    data_linear = slc_img[slc_mask,:]
    norms = np.mean(data_linear[:,~mask_noB0], axis = -1, keepdims = True)
    data_linear[np.abs(norms[:,0])>1e-5,:] = data_linear[np.abs(norms[:,0])>1e-5,:]/norms[np.abs(norms[:,0])>1e-5]
    n_voxels = data_linear.shape[0]
    data_shape = slc_img.shape
        
    #%%
    idx = 800
    plt.figure()
    plt.plot(data_linear[idx,:])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection = '3d')
    ax.scatter(bvecs[mask_shells[-2,:],0], bvecs[mask_shells[-2,:],1], bvecs[mask_shells[-2,:],2],
               c = data_linear[idx,:][mask_shells[-2,:]], cmap = 'turbo')
    plt.show()
    
    #%% computation of basis and projection matrix
    mask_bval = np.abs(bvals[mask_noB0]-bval)<100
    
    wig = sph.Wigner(L)
    Ymn = wig.sYlm(0, Rs_grid[mask_bval,:]) 
    n_sh = np.array([n for n in range(L+1) for m in range(-n,n+1)])
    proj_mat, c_proj_mat = shu.get_SH_proj_mat(L,None, n_sh, smooth = None, 
                                               cSH = Ymn, c_smooth = c_smooth)
    c_proj_mat = c_proj_mat.astype('complex64')
    #%%
    x = data_linear[:,mask_noB0][:,mask_bval]
    x0mean = x - np.mean(x,axis = -1, keepdims = True) #Remove the mean to avoid singularity at n = 0
    del x
    #%% Compute SH expansion
    cSH_x = (c_proj_mat @ x0mean.T).T
    
    
    #%% Load the direction estimations

    u,p = scipy.linalg.polar(full_img.affine[0:3, 0:3]) #TODO compute the rotation matrix from the quaternions stored in the header
    if(dir_est_type == 'CSD'):
        print('Loading CSD fod peaks...')
        csd_peaks_path = os.path.join(base_patient_path, *['diff','preproc','mri','mrtrix_fod_peaks.nii.gz'])
        # Load nifti
        csd_peaks_nifti = nib.load(csd_peaks_path)
        # Get data
        csd_peaks_raw = csd_peaks_nifti.get_fdata(dtype = np.float32)
        num_peaks = csd_peaks_raw.shape[-1]//3
        dir_est = csd_peaks_raw.reshape(*csd_peaks_raw.shape[0:3], num_peaks,3) #Last axis are the coordinates ; second to last is the peak index, so 3 peaks were computed
        norms_dir_est = np.linalg.norm(dir_est, axis = -1, keepdims = True)
        dir_est = dir_est/norms_dir_est 
        dir_est = dir_est @ u #Inverse rotation
        print('Loading diffusion data... Done!')
    elif(dir_est_type == 'MSMTCSD'):
        print('Loading MSMT CSD fod peaks...')
        msmt_peaks_path = os.path.join(base_patient_path, *['diff','preproc','mri','mrtrix_msmt_fod_peaks.nii.gz'])
        # Load nifti
        msmt_peaks_nifti = nib.load(msmt_peaks_path)

        # Get data
        msmt_peaks_raw = msmt_peaks_nifti.get_fdata(dtype = np.float32)
        num_peaks = msmt_peaks_raw.shape[-1]//3
        dir_est = msmt_peaks_raw.reshape(*msmt_peaks_raw.shape[0:3], num_peaks,3) #Last axis are the coordinates ; second to last is the peak index, so 3 peaks were computed
        norms_dir_est = np.linalg.norm(dir_est, axis = -1, keepdims = True)
        dir_est = dir_est/norms_dir_est
        dir_est = dir_est @ u #Inverse rotation
        # The peaks are ordered according to their norm
        msmt_num_peaks = np.sum(norms_dir_est.squeeze()>0.1, axis = -1) #Building a map of number of peaks
        print('Loading MSMT CSD fod peaks...Done')
        
        
        header = full_img.header
        header.dim = msmt_num_peaks.squeeze().shape
        nifti = nib.Nifti1Image(msmt_num_peaks.squeeze(), full_img.affine, header=header, dtype = 'int8')
        nib.save(nifti, os.path.join(save_path, '{0}_num_peaks.nii.gz').format(run_id))
        
    else:
        raise ValueError(f'Direction estimation type not recognized. Got {dir_est_type} but expected CSD or MSMTCSD')
    slc_dir_est = dir_est[tuple(slc+[slice(None), slice(None)])]                   
    dir_est_linear = slc_dir_est[slc_mask,:]
    
    #Num peaks
    if(fix_number_of_peaks):
        num_peaks = dir_est.shape[3]
    else:
        if(dir_est_type == 'MSMTCSD'):
            num_peaks = msmt_num_peaks[tuple(slc)][slc_mask]
    
        
    #%% Prepare the parallelization by dividing the linear data into chunks 
    nsamples = cSH_x.shape[0]
    chunksize = nsamples//num_processes
    print('Chunksize : ', chunksize)
    slc_chunks = []
    start = 0
    for k in range(num_processes):
        end = min(nsamples, start + chunksize)
        if(k==(num_processes-1) and end<nsamples):
            end = nsamples
        slc_current_chunk = slice(start,end)
        slc_chunks.append(slc_current_chunk)
        start = end
    
    
    #%% Prepare the logfile 
    logpath = os.path.join(save_path, 'Logs')
    if(not(os.path.exists(logpath))):
        os.makedirs(logpath)
    #%% Prepare the dask delayed objects
    futures = []
    #Prepare the function which will be applied to the chunks
    fun = lambda x:sbsd.ComputeCanonicalSignals(L, 
                                                x[0], 
                                                x[1],
                                                x[2],
                                                t_smooth, 
                                                verbosity = 0,
                                                store_residuals = store_residuals,
                                                logpath = x[3])
    
    for i,slc_chunk in enumerate(slc_chunks):
        log =  os.path.join(logpath,'logs_{0}.txt'.format(i))
        if(fix_number_of_peaks):
            fut = dask.delayed(fun)([cSH_x[slc_chunk,...], 
                                     num_peaks,
                                     dir_est_linear[slc_chunk,0:np.max(num_peaks),:],
                                     log])
        else:
            fut = dask.delayed(fun)([cSH_x[slc_chunk,...], 
                                     num_peaks[slc_chunk],
                                     dir_est_linear[slc_chunk,0:np.max(num_peaks),:],
                                     log])
        futures.append(fut)
      
    #%% Prepare the progress bar
    pb = ProgressBar()
    pb.register()
    

    #%% Run the estimation of SH
    start = time.time()
    print('Starting computations...')
    with dask.config.set(scheduler = 'processes', num_workers = num_processes):
        results = dask.compute(futures)
    print('Computations done')
    print('\n\n====== Time taken : ', time.time() - start)
    
    #%% Concatenate the results
    weights = np.concatenate([results[0][j][0] for j in range(num_processes)], axis = 0)
    Rus = np.concatenate([results[0][j][1] for j in range(num_processes)], axis = 0)
    if(store_residuals):
        residuals = np.concatenate([results[0][j][2] for j in range(num_processes)], axis = 0)
    #%% Save the results as nifti if needed
    output_weights = np.zeros((*data_shape[0:3],weights.shape[1], weights.shape[-1]), dtype = 'complex')
    output_weights[slc_mask.astype(bool),:] = weights
    
    output_residuals = np.zeros((*data_shape[0:3],residuals.shape[-1]), dtype = 'complex')
    output_residuals[slc_mask.astype(bool),:] = residuals
    
    output_num_peaks = np.zeros(data_shape[0:3], dtype = 'int8')
    output_num_peaks[slc_mask.astype(bool)] = num_peaks

    if(save_result_as_nifti):
        header = full_img.header
        header.dim = output_weights.shape
        nifti_weights = nib.Nifti1Image(output_weights.real, full_img.affine, header=header)
        nib.save(nifti_weights, os.path.join(save_path, '{0}_SHEST_weights_real.nii.gz').format(run_id))
        
        nifti_weights = nib.Nifti1Image(output_weights.imag, full_img.affine, header=header)
        nib.save(nifti_weights, os.path.join(save_path, '{0}_SHEST_weights_imag.nii.gz').format(run_id))
        
        nifti_weights = nib.Nifti1Image(np.abs(output_weights), full_img.affine, header=header)
        nib.save(nifti_weights, os.path.join(save_path, '{0}_SHEST_weights_modulus.nii.gz').format(run_id))
        if(store_residuals):
            header = full_img.header
            header.dim = output_residuals.shape
            
            nifti= nib.Nifti1Image(output_residuals.imag, full_img.affine, header=header)
            nib.save(nifti, os.path.join(save_path, '{0}_residuals_imag.nii.gz').format(run_id))
            
            nifti= nib.Nifti1Image(output_residuals.real, full_img.affine, header=header)
            nib.save(nifti, os.path.join(save_path, '{0}_residuals_real.nii.gz').format(run_id))
            
            norm_residuals = np.sum(np.sqrt(output_residuals.real**2 + output_residuals.imag**2), axis = -1)
            header = full_img.header
            header.dim = norm_residuals.shape
            nifti= nib.Nifti1Image(norm_residuals, full_img.affine, header=header)
            nib.save(nifti, os.path.join(save_path, '{0}_norm_residuals.nii.gz').format(run_id))
        if(not(fix_number_of_peaks)):
            header = full_img.header
            header.dim = output_num_peaks.squeeze().shape
            nifti = nib.Nifti1Image(output_num_peaks.squeeze(), full_img.affine, header=header, dtype = 'int8')
            nib.save(nifti, os.path.join(save_path, '{0}_num_peaks.nii.gz').format(run_id))

        

        
    
    #%% Store the metadata (parameters)
    
    metadata = {'L':L,
                'c_smooth':c_smooth,
                't_smooth':t_smooth,
                'bval':bval,
                'bval_idx':int(bval_idx),
                'dir_est_type':dir_est_type}
    
    with open(os.path.join(save_path, 'run_{0}.json'.format(run_id)), 'w') as f:
        json.dump(metadata, f)


