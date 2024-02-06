# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import math
import pickle

path_SBSD = "E:/github/SBSD_public/" #Put the path where you git cloned SBSD_public
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)     
 
from core import sh_utils as shu
from core import DataGenerator as DG

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
import dipy.reconst.dti as dti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

#%% Load data
base_data_path = os.path.join(path_SBSD, 'data')
data_run_id = 0
schem_suffix = '200' #Number of directions in the acquisition scheme
num_fasc = 2
nsamples = 100 #Data.data['nus'].shape[0]
for SNR in [100]:
    task_name = 'SNR' + str(SNR)
    print('\n\n************** Task name : {0} **************\n\n'.format(task_name))
    scheme_name = "01scheme_"+schem_suffix

    task_name = scheme_name + '_SNR'+str(SNR)
    Data = DG.DataLoader(base_data_path, 
                 task_name, 
                 data_run_id,
                 num_samples = nsamples,
                 num_fasc = num_fasc)
    
    N_unkn = Data.data['nus'].shape[1]
    crossangle_min = Data.data['parameters']['crossangle_min']
    #%%Load dictionary 
    base_dic_path = os.path.join(path_SBSD, "MC_simus")
    dic_name = "MC_simus.pkl"
    dictionary_path = os.path.join(base_dic_path, dic_name)
    with open(dictionary_path, 'rb') as f:
        full_dic = pickle.load(f)
    dic = full_dic[scheme_name]
    bvecs = dic['sch_mat'][:,0:3]
    bvals = dic['bvals']
    ubvals = np.unique(bvals)
    idx_nob0 = bvals>1e2
    xyz = bvecs[idx_nob0,:]
    D = dic['dictionary']
    
    #Choose the bval
    bval = ubvals[-1]
    mask_bval = np.abs(bvals[idx_nob0] - bval)<5e3
    
    
    
    #%% Preparing CSD
    tol = 500
    gtab = gradient_table(bvals / 1e6, bvecs)
    ubvals = unique_bvals_tolerance(gtab.bvals, tol) #List of possible values for bvals
    
    selected_bvals = np.abs(bvals - bval) < tol*1e6 / 1e6 
    selected_bvals[0] = True #To have one b0 value
    gtab_single_shell = gradient_table(bvals[selected_bvals], bvecs[selected_bvals,:])
    
    tenmodel = dti.TensorModel(gtab_single_shell)
    tenfit = tenmodel.fit(np.mean(D[selected_bvals,:], axis = 1).T) 
    
    #%%
    evals = tenfit.evals
    csd_response = [evals,1]
    csd_model = ConstrainedSphericalDeconvModel(gtab_single_shell,
                                          csd_response,
                                          sh_order=8)
    odf_sphere = get_sphere('repulsion724') #Sphere for regularization
    
    
    
    #%%
    x0 = np.max(Data.data['DWI_noisy'][:,~idx_nob0,], axis = 1, keepdims = True)
    x = Data.data['DWI_noisy'][:,idx_nob0][:,mask_bval]/x0
    
        
    
        
    #%%
    csd_estimated_us = np.zeros((nsamples, N_unkn,3), dtype = 'float64')
    sh_order = 8
    min_separation_angle = 25
    for j in tqdm(range(nsamples)): 
        current_first = 0    
        
        # Perform CSD
        y = np.concatenate((x0[j:j+1,0], x[j,:]))
        peaks = peaks_from_model(model=csd_model,
                                 data=y,
                                 sphere=odf_sphere,
                                 relative_peak_threshold=0.15,
                                 min_separation_angle=min_separation_angle,
                                 sh_order=sh_order,
                                 npeaks=2)
        
        csd_estimated_us[j,:,:] = peaks.peak_dirs
            
    
            
    
    #%% Check CSD results
    norms = np.linalg.norm(csd_estimated_us[:,:,:], axis = 2)
    prop_no_scnd_peak = np.sum(norms<0.1)/nsamples
    
    
    print('Proportion of samples for which the CSD failed to detect the second peak : ',prop_no_scnd_peak)
    gt_us = Data.data['orientations'][0:nsamples,:]
    
    csd_matched_estimated_us = shu.match_peaks(gt_us[:,:,:], csd_estimated_us[:,:,:])  
    csd_dots = np.abs(np.sum(gt_us[:,:,:] * csd_matched_estimated_us[:,:,:], axis = 2))
    
    # csd_no_swap_dots = np.abs(np.sum(gt_us[:,:,:] * csd_estimated_us[:,:,:], axis = 2))
    # csd_swapped_dots = np.abs(np.sum(np.flip(gt_us[:,:,:], axis =1) * csd_estimated_us[:,:,:], 
    #                              axis = 2))
    # csd_dots = np.where(csd_no_swap_dots>csd_swapped_dots,csd_no_swap_dots,csd_swapped_dots)
    
    
    csd_mask = csd_dots>1
    csd_dots[csd_mask] = 1
    
    report = '\n\n Task Name : ' + str(task_name)
    report = report + '\n SH_order : ' + str(sh_order)
    report = report + '\n Min sepration angle : ' + str(min_separation_angle)
    msg = '\n\n======== Results with CSD'
    report = report + msg
    print(msg)
    report = report + '\nTASK NAME : ' + task_name
    if(np.sum(csd_mask)>0):
        msg = 'Found {0} samples with dot >1, with mean {1}'.format(np.sum(csd_mask), 
                                                                np.mean(csd_dots[csd_mask]))
        report = report + '\n' + msg
        print(msg)
    csd_ang_errors = np.arccos(csd_dots)*180/np.pi
    
    msg = 'Mean dot : '+str( np.mean(csd_dots))
    report = report + '\n' + msg
    print(msg)
    
    msg = 'Mean angular errors : ' + str(np.mean(csd_ang_errors))
    report = report + '\n' + msg
    print(msg)
    
    msg = 'Median angular errors : '+ str(np.median(csd_ang_errors))
    report = report + '\n' + msg
    print(msg)
    
    msg = 'Proportion below 5° error : ' + str(np.sum(csd_ang_errors<5)/csd_ang_errors.shape[0]/2)
    report = report + '\n' + msg
    print(msg)
    
    msg = 'Proportion below 10° error : ' + str(np.sum(csd_ang_errors<10)/csd_ang_errors.shape[0]/2)
    report = report + '\n' + msg
    print(msg)
    
    msg = 'Proportion below 20° error : ' + str(np.sum(csd_ang_errors<20)/csd_ang_errors.shape[0]/2)
    report = report + '\n' + msg
    print(msg)
    

    
    uid = 0
    save_path = "E:\github\SBSD_private\Reports\RotationEstimateMC_CSD"
    str_bval = str(math.trunc(bval*1e-6))
    name = 'report_{0}.txt'.format(uid)
    report_path = os.path.join(save_path,name )
    while(os.path.exists(report_path)):
        uid = uid+1
        name = 'csd_report_{0}.txt'.format(uid)
        report_path = os.path.join(save_path,name )
    with open(report_path, 'w') as f:
        f.write(report)