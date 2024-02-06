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
    path_SBSD = "E:/github/SBSD_public/" #Put the path where you git cloned SBSD_public
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
    cache_path = os.path.join(path_SBSD, "CachedRotationDictionary")
    patient_id = "1007"
    run_id = '07'
    
    #%% Build paths
    base_patient_path = os.path.join(base_hcp_path, *['MGH_{0}_all'.format(patient_id),
    'mgh_{0}'.format(patient_id)])
    save_path = os.path.join(base_patient_path,*['diff','preproc','mri','SBSD_{0}'.format(run_id)])
    if(not(os.path.exists(save_path))):
        os.makedirs(save_path)
    nifti_path = os.path.join(base_patient_path, *['diff','preproc','mri','diff_preproc.nii.gz'])
    #%% Parameters
    L = 8 #Degree for the computation of the expansion
    c_smooth = 1e-4 #For Laplace Beltrami regularization in the computation of the projection matrix
    t_smooth = 1e-4 #For tikhonov regularization in LS solving in OMP
    
    #Rotations dictionary of the following degrees will be used
    Ns = np.array([2,4,6,8]) #4 is usually the best choice
    li_slc_Ns = [slice(n**2,(n+1)**2) for n in Ns] #To slice the SH expansion of the DWI signal
    
    bval = 5000
    save_result_as_nifti = True
    
    # Slice on which the algortihm will be performed : None everyhwere -> perform on the complete volume (masked by a brain mask)
    slc = [slice(None),slice(None),slice(None)]  # Create a list of slices for each dimension
    #Slice for vizualisation
    visu_slc = [slice(None),slice(80,81),  slice(None)]  # Create a list of slices for each dimension
    visu_slc_diff = visu_slc + [slice(0,1)] #Visualization of first diff volume
    
    #Parameters of the SBSD algorithm
    max_n_peaks = 2
    epsilon = None
    fixed_num_peaks = 2
    
    #Parallelization
    num_processes = 50
    
    #%% Load data
    print('Loading diffusion data...')
    
    # Load nifti
    full_img = nib.load(nifti_path)
    
    # Get data
    img = full_img.get_fdata(dtype = np.float32)
    print('Loading diffusion data... Done!')
    
    
    #%% Load mask (dwi)
    # Load masks
    mask_name = "MNI_mask_{0}.nii.gz".format(patient_id)
    mask_path = os.path.join(base_hcp_path, *['brain_masks', 'MNI', mask_name])
    mask_nifti = nib.load(mask_path)
    mask = mask_nifti.get_fdata(dtype = np.float32).astype('bool')
    
    #%% Check segmentation 
    plt.figure()
    plt.title('Mask')
    mp = plt.matshow(img[tuple(visu_slc_diff)].squeeze()*mask[tuple(visu_slc)].squeeze(), cmap = 'magma', fignum = 0)
    plt.colorbar(mp)
    plt.show()
    
    
    plt.figure()
    mp = plt.matshow(img[tuple(visu_slc_diff)].squeeze(), cmap = 'magma', fignum = 0)
    plt.colorbar(mp)
    plt.show()
    
    
    #%% Get the bvals and bvecs
    bvals_path =os.path.join(*['E:\\',
                  'dataHCP',f'MGH_{patient_id}_all',f'mgh_{patient_id}',
                  'diff','preproc', 'bvals.txt'])
    bvecs_path =os.path.join(*['E:\\',
                  'dataHCP',f'MGH_{patient_id}_all',f'mgh_{patient_id}',
                  'diff','preproc', 'bvecs_moco_norm.txt'])
    
    bvals = np.loadtxt(bvals_path)
    ubvals = unique_bvals_tolerance(bvals, tol = 70)
    bval_idx = np.argmin(np.abs(ubvals-bval))
    bvecs = np.loadtxt(bvecs_path )
    mask_noB0 = bvals>100
    xyz = bvecs[mask_noB0,:]
    
    #Compute maskes for isolating shells
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
    n_voxels = data_linear.shape[0]
    data_shape = slc_img.shape
    
    
    #%% Normalize data with values at b = 0
    norms = np.max(data_linear[:,~mask_noB0],axis = 1, keepdims = True)
    data_linear[norms[:,0]>1e-6,:] = data_linear[norms[:,0]>1e-6,:]/norms[norms[:,0]>1e-6,:]
    
    #%%
    idx = 800
    plt.figure()
    plt.plot(data_linear[idx,:])
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection = '3d')
    ax.scatter(bvecs[mask_shells[-2,:],0], bvecs[mask_shells[-2,:],1], bvecs[mask_shells[-2,:],2],
               c = data_linear[idx,:][mask_shells[-2,:]], cmap = 'Greys')
    plt.show()
    
    #%% computation of basis and projection matrix
    mask_bval = np.abs(bvals[mask_noB0]-bval)<100
    
    wig = sph.Wigner(L)
    Ymn = wig.sYlm(0, Rs_grid[mask_bval,:]) 
    n_sh = np.array([n for n in range(L+1) for m in range(-n,n+1)])
    proj_mat, c_proj_mat = shu.get_SH_proj_mat(L,None, n_sh, smooth = None, 
                                               cSH = Ymn, c_smooth = c_smooth)
    
    #%%
    x = data_linear[:,mask_noB0][:,mask_bval]
    
    #%% Compute SH expansion
    cSH_x = (c_proj_mat @ x.T).T
    
    
    #%% Load dictionary of rotations
    DNs = []
    li_xyz_rots = []
    nangles = 100
    dr = 0.02

    for k,N in enumerate(Ns):
        name_D = 'D_nangles_{0}_dangle_{1}_N_{2}.npy'.format(nangles,dr,N)
        name_xyz = 'xyz_nangles_{0}_dangle_{1}_N_{2}.npy'.format(nangles,dr,N)
        DN = np.load(os.path.join(cache_path, name_D))
        DNs.append(DN)
        
        xyz_rot = np.load(os.path.join(cache_path, name_xyz))
        li_xyz_rots.append(xyz_rot)
        
    #%%
    for k,N in enumerate(Ns):
        xyz_rot = li_xyz_rots[k]
        fig = plt.figure()
        fig.suptitle('Degree : {0}'.format( N))
        ax = fig.add_subplot(1,1,1, projection = '3d')
        ax.scatter(xyz_rot[:,0], xyz_rot[:,1],xyz_rot[:,2], s = 1)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.show()
        
    #%% Prepare the parallelization by dividing the linear data into chunks 

    nsamples = cSH_x.shape[0]
    chunksize = nsamples//num_processes
    print('Chunksize : ', chunksize)
    print('Total number of samples : ', nsamples)
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
    fun = lambda x:sbsd.run_SBSD(x[0], 
                                    DNs, 
                                    li_xyz_rots,
                                    Ns = Ns, 
                                    epsilon = epsilon,
                                    fixed_num_peaks = fixed_num_peaks,
                                    max_n_peaks = max_n_peaks,
                                    t_smooth = t_smooth,
                                    verbosity = 0,
                                    logpath = x[1]) #Verbosity = 0 is recommended when using parallelization
    
    for i,slc_chunk in enumerate(slc_chunks):
        log =  os.path.join(logpath,'logs_{0}.txt'.format(i))
        fut = dask.delayed(fun)([cSH_x[slc_chunk,], log])
        futures.append(fut)
    #%% Prepare the progress bar
    pb = ProgressBar()
    pb.register()
    
    #%% Run the SBSD algorithm
    start = time.time()
    print('Starting computations...')
    with dask.config.set(scheduler = 'processes', num_workers = num_processes):
        results = dask.compute(futures)
    print('Computations done')
    running_time = time.time() - start
    print('\n\n====== Time taken : ', running_time)
    
    #%% Concatenate the results
    Aks = np.concatenate([results[0][j][0] for j in range(num_processes)], axis = 0)
    Wks = np.concatenate([results[0][j][1] for j in range(num_processes)], axis = 0)
    Cks = np.concatenate([results[0][j][2] for j in range(num_processes)], axis = 0)
    estimated_us = np.concatenate([results[0][j][3] for j in range(num_processes)], axis = 0)
    estimated_s  = [np.concatenate([results[0][j][4][id_degree] for j in range(num_processes)], axis = 0) for id_degree in range(len(Ns))]
    rks =  [np.concatenate([results[0][j][5][id_degree] for j in range(num_processes)], axis = 0) for id_degree in range(len(Ns))]
    n_peaks  = np.concatenate([results[0][j][6] for j in range(num_processes)], axis = 0)
    
    #%% Put back into right shape
    output_Wks = np.zeros((*data_shape[0:3],len(Ns), Aks.shape[-1]), dtype = 'complex')
    output_Wks[slc_mask.astype(bool),:] = Wks
    
    output_npeaks = np.zeros((*data_shape[0:3],len(Ns)), dtype = 'int32')
    output_npeaks[slc_mask.astype(bool),:] = n_peaks
    
    output_Cks = np.zeros((*data_shape[0:3],len(Ns), Aks.shape[-1]), dtype = 'complex')
    output_Cks[slc_mask.astype(bool),:] = Cks
    
    output_estimated_us = np.zeros((*data_shape[0:3],len(Ns), Aks.shape[-1], 3), dtype = 'float64')
    output_estimated_us[slc_mask.astype(bool),:,:] = estimated_us
    
    #%% Save the results as nifti if needed
    if(save_result_as_nifti):
        nifti_Wks = nib.Nifti1Image(np.abs(output_Wks), full_img.affine, header=full_img.header)
        nib.save(nifti_Wks, os.path.join(save_path, '{0}_SBSD_Wks_modulus.nii.gz'.format(run_id)))
        
        nifti_Wks = nib.Nifti1Image(output_Wks.real, full_img.affine, header=full_img.header)
        nib.save(nifti_Wks, os.path.join(save_path, '{0}_SBSD_Wks_real.nii.gz').format(run_id))
        
        nifti_Wks = nib.Nifti1Image(output_Wks.imag, full_img.affine, header=full_img.header)
        nib.save(nifti_Wks, os.path.join(save_path, '{0}_SBSD_Wks_imag.nii.gz').format(run_id))
        
        nifti_npeaks = nib.Nifti1Image(output_npeaks, full_img.affine, header=full_img.header)
        nib.save(nifti_Wks, os.path.join(save_path, '{0}_SBSD_npeaks.nii.gz').format(run_id))
        
        nifti_Cks = nib.Nifti1Image(np.abs(output_Cks), full_img.affine, header=full_img.header)
        nib.save(nifti_Cks, os.path.join(save_path, '{0}_SBSD_Cks_modulus.nii.gz').format(run_id))
        
        nifti_estimated_us = nib.Nifti1Image(output_estimated_us, full_img.affine, header=full_img.header)
        nib.save(nifti_estimated_us, os.path.join(save_path, '{0}_SBSD_estimated_us.nii.gz').format(run_id))

        
    
    #%% Store the metadata (parameters)
    
    metadata = {'L':L,
                'c_smooth':c_smooth,
                't_smooth':t_smooth,
                'Ns':Ns.tolist(),
                'bval':bval,
                'bval_idx':int(bval_idx),
                'nangles':nangles,
                'dr':dr,
                'fixed_num_peaks':fixed_num_peaks,
                'max_n_peaks':max_n_peaks,
                'espilon':epsilon,
                'total_running_time':running_time}
    
    with open(os.path.join(save_path, 'run_{0}.json'.format(run_id)), 'w') as f:
        json.dump(metadata, f)


