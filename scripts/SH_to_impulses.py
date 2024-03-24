# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import spherical as sph
from tqdm import tqdm
path_SBSD = "E:/github/SBSD_public_review/"
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)   
from core import sh_utils as shu
import json

# This script reconstructs impulses from estimated SH Expansions

#%% Paths
results_path = os.path.join("") # Path where the SH expansions are located
all_results = {}
li_bvals = [2000, 3000, 3999, 5000, 9999]
li_bvals_str = [2000, 3000, 4000, 5000, 10000]
li_schem_suffix = [100, 120, 130, 150,200,300]
li_SNR_suffix = [100, 50,40,30,20,10]
num_fasc = 2

#%% Load all canonical signals
base_dic_path = os.path.join(path_SBSD, "MC_simus")
dic_name = "MC_simus.pkl"
dictionary_path = os.path.join(base_dic_path, dic_name)
with open(dictionary_path, 'rb') as f:
    full_dic = pickle.load(f)
    

#%% Load results in a dictionary
rep_ids = {}
for bval in li_bvals:
    rep_id = "GT_10"
    bval_key = 'bval_'+str(bval)
    rep_ids[bval_key] = rep_id
    
#%%
print('\n\n\n****************** LOADING DATA ******************')
for _,schem_suffix in enumerate(li_schem_suffix):
    scheme_key = 'scheme_'+str(schem_suffix)
    all_results[scheme_key] = {}
    for _,SNR_suffix in enumerate(li_SNR_suffix):
        SNR_key = 'SNR_'+str(SNR_suffix)
        all_results[scheme_key][SNR_key] = {}

        for _,bval in enumerate(li_bvals):
            bval_key = 'bval_'+str(bval)
            rep_id = rep_ids[bval_key]
            scheme_name = "01scheme_"+ str(schem_suffix)
            task_name = scheme_name + '_SNR'+ str(SNR_suffix)
            results_name ='rep_' + rep_id + '_' + task_name + "_" +"_bval_" + str(bval)
            
            with open(os.path.join(results_path, results_name + '.json'), 'rb') as f:
                metadata = json.load(f)
            with open(os.path.join(results_path, results_name + '.pickle'), 'rb') as f:
                data = pickle.load(f)
            
            all_results[scheme_key][SNR_key][bval_key] = {}
            all_results[scheme_key][SNR_key][bval_key]['metadata'] = metadata
            all_results[scheme_key][SNR_key][bval_key]['data'] = data
            print('\n============ KEYS : ', [scheme_key, SNR_key, bval_key])
            print(metadata)

            

#%% Load the results for easy access and plotting
li_degrees = [0,2,4,6,8]
nsamples = 20000

all_gt_weights = {}
all_weights = {}
all_errors = {}
all_rel_errors = {}
li_dics = [all_gt_weights,all_weights,all_errors,all_rel_errors]
for j,schem_suffix in enumerate(li_schem_suffix):
    scheme_key = 'scheme_'+str(schem_suffix)
    for dic in li_dics:
        dic[scheme_key] = {}
    for k,SNR_suffix in enumerate(li_SNR_suffix):
        SNR_key = 'SNR_'+str(SNR_suffix)
        for dic in li_dics:
            dic[scheme_key][SNR_key] = {}
        for i,bval in enumerate(li_bvals):
            bval_key = 'bval_'+str(bval)
            all_gt_weights[scheme_key][SNR_key][bval_key] = all_results[scheme_key][SNR_key][bval_key]['data']['gt_weights'][:,:,:]
            all_weights[scheme_key][SNR_key][bval_key] = all_results[scheme_key][SNR_key][bval_key]['data']['est_weights'][:,:,:]
            all_errors[scheme_key][SNR_key][bval_key] = np.abs(all_gt_weights[scheme_key][SNR_key][bval_key] - all_weights[scheme_key][SNR_key][bval_key])
            all_rel_errors[scheme_key][SNR_key][bval_key]  = all_errors[scheme_key][SNR_key][bval_key]/np.abs(all_gt_weights[scheme_key][SNR_key][bval_key])
 

#%% Reconstruct the impulse responses

all_impulses = {}
all_gt_impulses = {}
all_impulses['bvals'] = li_bvals
all_gt_impulses['bvals'] = li_bvals
for j,schem_suffix in tqdm(enumerate(li_schem_suffix)):
    scheme_key = 'scheme_'+str(schem_suffix)
    dic = full_dic['01'+scheme_key]
    mask_bvals,ubvals,mask_nob0,all_idx_sort = shu.build_mask(dic)
    bvecs = dic['sch_mat'][:,0:3][mask_nob0,:]
    Rs_grid,theta,phi,r,xyz = shu.build_Rs(dic)
    all_impulses[scheme_key] = {}
    all_gt_impulses[scheme_key] = {}
    for k,SNR_suffix in enumerate(li_SNR_suffix):
        SNR_key = 'SNR_'+str(SNR_suffix)
        all_impulses[scheme_key][SNR_key] = {} #np.zeros((nsamples, len(li_bvals),num_fasc, schem_suffix), dtype = 'float32')
        all_gt_impulses[scheme_key][SNR_key] = {} #np.zeros((nsamples, len(li_bvals),num_fasc, schem_suffix), dtype = 'float32')
        for i,bval in enumerate(li_bvals):
            bval_key = 'bval_'+str(bval)
            xyz = bvecs[mask_bvals[i,:],:]
            z = xyz[:,2][all_idx_sort[i,:]]
            
            
            all_impulses[scheme_key][SNR_key][bval_key] = np.zeros((nsamples, num_fasc, schem_suffix), dtype = 'float32')
            all_gt_impulses[scheme_key][SNR_key][bval_key] = np.zeros((nsamples, num_fasc, schem_suffix), dtype = 'float32')
            
            mask_bval = mask_bvals[i,:]
            weights = all_weights[scheme_key][SNR_key][bval_key]
            gt_weights = all_gt_weights[scheme_key][SNR_key][bval_key]
            #Compute basis matrix and projection matrix
            L = (weights.shape[-1]-1)*2 # There are L//2 + 1 SH coefficients for each estimated impulse

            wig = sph.Wigner(L)
            Ymn = wig.sYlm(0, Rs_grid[mask_bval,:]) 
            mask_h0 = np.array([n%2==0 and m==0 for n in range(L+1) for m in range(-n,n+1)])
            
            # Reconstruct estimated impulses
            sh_expansions = np.zeros((*weights.shape[0:2], Ymn.shape[1]), dtype = 'complex')
            sh_expansions[:,:,mask_h0] = weights
            impulses = (sh_expansions @ Ymn.T).real
            
            if(schem_suffix==120):
                ws = 12
            elif(schem_suffix == 130):
                ws = 12
            else:
                ws = schem_suffix//10
            impulses = shu.pp_impulses(impulses[:,:,all_idx_sort[i,:]], z, ws = ws, start_z = 0.9)
            mask_corr = np.any(impulses<0, axis = -1)
            impulses[mask_corr,:] = impulses[mask_corr,:] - np.min(impulses[mask_corr,:], axis = -1, keepdims = True)
            invert_permut = np.argsort(all_idx_sort[i,:])
            impulses = impulses[:,:,invert_permut] #So that the values are ordered like in the other impulses
            
            
            # Reconstruct gt impulses
            gt_sh_expansions = np.zeros((*weights.shape[0:2], Ymn.shape[1]), dtype = 'complex')
            gt_sh_expansions[:,:,mask_h0] = gt_weights
            gt_impulses = gt_sh_expansions @ Ymn.T
            
            #Store the reconstructed impulses in the dictionary
            all_impulses[scheme_key][SNR_key][bval_key][:,:,:] = impulses[:,:,:]
            all_gt_impulses[scheme_key][SNR_key][bval_key][:,:,:] = gt_impulses[:,:,:]
            
            
#%% Store the impulses
save_path = os.path.join(results_path, 'reconstructed_impulses')
if(not(os.path.exists(save_path))):
    os.makedirs(save_path)
    
with open(os.path.join(save_path, 'recon_impulses.pkl'), 'wb') as file:
    pickle.dump(all_impulses,file)
    
with open(os.path.join(save_path, 'recon_gt_impulses.pkl'), 'wb') as file:
    pickle.dump(all_gt_impulses,file)
    
#%% Some sanity checks
sample = np.random.randint(nsamples)
SNR_key = 'SNR_100'
scheme_key = 'scheme_100'
dic = full_dic['01'+scheme_key]
mask_bvals,ubvals,mask_nob0,all_idx_sort = shu.build_mask(dic)
Rs_grid,theta,phi,r,xyz = shu.build_Rs(dic)
imp = all_impulses[scheme_key][SNR_key]
gt_imp = all_gt_impulses[scheme_key][SNR_key]
fig,axs = plt.subplots(2,1, squeeze = False)
fig.suptitle(f'Sample : {sample}')
for j,bval in enumerate(li_bvals):
    bval_key = 'bval_'+str(bval)
    for k in range(num_fasc):
        id_sort = all_idx_sort[j,:]
        mask_bval = mask_bvals[j,:]
        x = xyz[mask_bval,2][id_sort]+2.05*j

        axs[k,0].plot(x,gt_imp[bval_key][sample,k, :][id_sort].real, '-k', ms = 1)
        axs[k,0].plot(x,imp[bval_key][sample,k, :][id_sort].real, '--g', ms = 2)
        # axs[k,2].plot(x,C[mask_noB0,gt_idx[k]][mask_bval][id_sort], '-c')

plt.show()





