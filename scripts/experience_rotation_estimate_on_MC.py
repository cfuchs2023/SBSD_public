# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pickle

path_SBSD = "E:/github/SBSD_public/" #Put the path where you git cloned SBSD_public
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)   
 

from core import sh_utils as shu
from core import DataGenerator as DG
import spherical as sph
import quaternionic as qua
from core import SBSD as sbsd
import json

#%% Parameters
#Where the reports and results will be stored
save_path = os.path.join(path_SBSD, "Reports","RotationEstimationMC_SBSD")

L = 8 #Degree for the computation of the expansion of the signal
N_unkn = 2 #Will be used for estiamting a fixed number of peaks
c_smooth = 1e-4 #Smooth factor for the computation of the projection matrix over SH
t_smooth = 1e-6 #For tikhonov regularization in LS solving in OMP

#Rotations dictionary of the following degrees will be used to estimate orientations : there are len(Ns) estimations (and different number of peaks can happen at different degree)
Ns = np.array([k for k in range(3,L+1) if k%2==0])
Ns = np.array([6])
li_slc_Ns = [slice(n**2,(n+1)**2) for n in Ns]

#Number of samples for each setting of (Number of directions, SNR)
nsamples = 2000

#Choose the bval
bval_idx = 3 #Choose the bval : 1,2,3,4,5 = 1000, 2000, 3000, 4000, 5000 ; 6 : 10000, 7 = 13000

#Parameters for the dictionary of U^n_{u_k}
nangles = 100
dangle = 0.02
#%% Loop over settings of (Number of directions, SNR)
base_data_path = os.path.join(path_SBSD, 'data')
for schem_suffix in ['200']:
    for SNR_suffix in ['100']:
        scheme_name = "01scheme_"+schem_suffix
        task_name = scheme_name + '_SNR'+SNR_suffix
        data_run_id = 0
        
        print('\n\n ================ New Computation ================')
        print('Scheme : ', scheme_name)
        print('SNR : ', SNR_suffix)
        

        Data = DG.DataLoader(base_data_path, 
                     task_name, 
                     data_run_id,
                     num_samples = nsamples,
                         num_fasc = 2)
        
        rep_id = "00"
        FORCE_OVERWRITE = True
        #%%
        base_dic_path = os.path.join(path_SBSD, "MC_simus")
        dic_name = "MC_simus.pkl"
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
        bval = ubvals[bval_idx]
        print('Bval (m/s^2) : ',bval*1e-6)
        mask_bval = np.abs(bvals[mask_noB0] - bval)<100
        xyz = bvecs[mask_noB0,:]
        
        #Compute spherical corrdinates of the grid, and corresponding quaternions
        theta,phi,r = shu.Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
        grid = np.zeros((theta.shape[0],2))
        grid[:,0] = theta[:]
        grid[:,1] = phi[:]
        Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
        xyz = shu.Sph2Cart(theta,phi,1)
        
        
        
        #%% Compute basis matrix and projection matrix
        wig = sph.Wigner(L)
        Ymn = wig.sYlm(0, Rs_grid[mask_bval,:]) 
        n_sh = np.array([n for n in range(L+1) for m in range(-n,n+1)])
        proj_mat, c_proj_mat = shu.get_SH_proj_mat(L,None, n_sh, smooth = None, 
                                                   cSH = Ymn, c_smooth = c_smooth)
        
        #%% Load dictionary of rotations
        base_cache_path = os.path.join(path_SBSD, "CachedRotationDictionary")
        DNs, li_xyz_rots = sbsd.LoadRotDic(Ns, dangle, nangles, base_cache_path)
        
        DNs_metadata = {
            'Ns':Ns.tolist(),
            'dangle':dangle,
            'nangles':nangles,
            'base_cache_path':base_cache_path
            
            }
        
        #%% Normalize data with values at b = 0
        x0 = np.max(Data.data['DWI_noisy'][:,~mask_noB0,], axis = 1, keepdims = True)
        x = Data.data['DWI_noisy'][:,mask_noB0][:,mask_bval]/x0
        x_noiseless = Data.data['DWI'][:,mask_noB0][:,mask_bval]/x0
        
        
        #%% Compute SH expansion of the data
        cSH_x = (c_proj_mat @ x.T).T
        recon_x = (Ymn @ cSH_x.T).T
        
        
        #%% Run the SBSD algorithm
        fixed_num_peaks = 2
        Aks, Wks, Cks, estimated_us, estimated_s, rks, num_peaks = sbsd.run_SBSD(cSH_x, 
                                                 DNs, 
                                                 li_xyz_rots,
                                                 Ns = Ns, 
                                                 epsilon = None,
                                                 fixed_num_peaks = fixed_num_peaks,
                                                 max_n_peaks = 2,
                                                 t_smooth = t_smooth)
        
        
        
        #%% Check orientations errors
        #SHape of estimated_us : (nsamples, N_unknowns, Number of Ns, Number of lambdas, 3)
        run_id = 'rep_' + rep_id + '_' + task_name + "_" +"_bval_" + str(math.trunc(bval*1e-6))
        if(not(os.path.exists(save_path))):
            os.makedirs(save_path)
        
        
        
        
        data_results_by_degree = {}
        gt_us = Data.data['orientations'][:,:]
        all_ang_errors = np.zeros((nsamples, len(Ns), N_unkn), dtype = 'float64')
        save_results = False
        plot_hists = True
        crossangle_min = Data.data['parameters']['crossangle_min']
        report = 'REPORT\n\n'
        report = report + '\nb-val : ' + str(math.trunc(bval*1e-6))
        report = report + '\nL : ' + str(L)
        report = report + '\n c_smooth : ' + str(c_smooth)
        report = report + '\n t_smooth : ' + str(t_smooth)
        report = report + '\nnsamples : '+ str(nsamples)
        report = report + '\nCross Angle min (°) : ' + str(crossangle_min)
        report = report + '\nnagles : ' + str(nangles)
        report = report + '\ndangle : ' + str(dangle)
        report = report + 'TASK NAME : ' + task_name
        for k,n in enumerate(Ns):      
        
            data_results_by_degree['Degree'+str(n)] = {}
            matched_estimated_us = shu.match_peaks(gt_us[:,:,:], estimated_us[:,k,:,:])  
            dots = np.abs(np.sum(gt_us[:,:,:] * matched_estimated_us[:,:,:], axis = 2))
            mask = dots>1 #To avoid bugs : sometimes, the dots can be slightly > 1 (1+1e-7,8,9)
            dots[mask] = 1
            msg = '\n\n======== Results with degree {0}'.format(n)
            report = report + msg
            print(msg)
            
            if(np.sum(mask)>0):
                msg = 'Found {0} samples with dot >1, with mean {1}'.format(np.sum(mask), 
                                                                        np.mean(dots[mask]))
                report = report + '\n' + msg
                print(msg)
            ang_errors = np.arccos(dots)*180/np.pi
            
            msg = 'Mean dot : '+str( np.mean(dots, axis = 0))
            report = report + '\n' + msg
            print(msg)
            
            
            print('Sanity check : Mean absolute imaginary part of WKs : ', np.mean(np.abs(Wks[:,k,:].imag)))

            
            mean_ang_errors = np.mean(ang_errors, axis = 0)
            msg = 'Mean angular errors : ' + str(mean_ang_errors)
            report = report + '\n' + msg
            print(msg)
            
            median_ang_errors = np.median(ang_errors, axis =0)
            msg = 'Median angular errors : '+ str(median_ang_errors)
            report = report + '\n' + msg
            print(msg)
            
            prop_5 = np.sum(ang_errors<5, axis = 0)/ang_errors.shape[0]
            msg = 'Proportion below 5° error : ' + str(prop_5)
            report = report + '\n' + msg
            print(msg)
            
            prop_10 = np.sum(ang_errors<10, axis = 0)/ang_errors.shape[0]
            msg = 'Proportion below 10° error : ' + str(prop_10)
            report = report + '\n' + msg
            print(msg)
            
            print('\n* Metrics over both population *')
            print('Mean dot : ', np.mean(dots))
            print('Proportion below 5° error : ', 
                  np.sum(ang_errors<5)/(N_unkn*ang_errors.shape[0]))
            print('Proportion below 10° error : ', 
                  np.sum(ang_errors<10)/(N_unkn*ang_errors.shape[0]))
            print('Mean angular errors : ', 
                  np.mean(ang_errors))
            print('Median angular errors : ', np.median(ang_errors))
                        
            data_results_by_degree['Degree'+str(n)] = {    
                'mean_ang_error':np.mean(ang_errors, axis = 0).tolist(),
                'median_ang_error':np.median(ang_errors, axis = 0).tolist(),
                'prop_below_5':(np.sum(ang_errors<5, axis = 0)/ang_errors.shape[0]).tolist(),
                'prop_below_10':(np.sum(ang_errors<10, axis = 0)/ang_errors.shape[0]).tolist(),
                'both_median_ang_error':np.median(ang_errors)
                
                }
            
            all_ang_errors[:,k,:] = ang_errors
                
                
        #%% Write report
        
        if(not(os.path.exists(save_path))):
            os.makedirs(save_path)
            
            
        data = {
            'task_name':task_name,
            'dic_path':dictionary_path,
            'scheme_name':scheme_name,
            'base_data_path':scheme_name,
            'L':L,
            'bval':bval,
            'c_smooth':c_smooth,
            't_smooth':t_smooth,
            'DNs_metadata':DNs_metadata,
            'results':data_results_by_degree,
            }
        
        
        if(not(FORCE_OVERWRITE) and os.path.exists(os.path.join(save_path, run_id+'.json'))):
            raise RuntimeError('File already exists.')
            
        with open(os.path.join(save_path, run_id+'.json'), 'w') as f:
            json.dump(data, f, indent = 2)
            
        #%% Store results
        #Aks, Wks, Cks, estimated_us, estimated_s, rks, num_peaks
        data = {'Aks':Aks,
                'Wks':Wks,
                'Cks':Cks,
                'estimated_us':estimated_us,
                'rks':rks,
                'num_peaks':num_peaks
                }
        
        with open(os.path.join(save_path, run_id+'.pickle'), 'wb') as f:
            pickle.dump(data, f)

