# -*- coding: utf-8 -*-
import numpy 
import os
import numpy as np
import pickle
import sys

path_SBSD = "E:/github/SBSD_public_review/" #Put the path where you git cloned SBSD_public
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)   
    
from core import DataGenerator as DG


#Notations and names :
    # atom = one canonical signal simulated with MC-DC
    # dictionary = Set of canonical signals simulated with MC-DC
    # Sample : generated data point
    # sch : acquisition scheme 

#%% Parameters
num_samples = 20000
num_fasc = 2 #Number of crossing fascicles

#%%
base_sch_path = os.path.join(path_SBSD, "MC_simus")
base_dic_path = base_sch_path
out_path = os.path.join(path_SBSD, "BigData")
if(not(os.path.exists(out_path))):
    os.makedirs(out_path)

for schem_suffix in ['300','200','150','140', '130', '120', '110', '100']:
    scheme_name = "01scheme_"+schem_suffix
    print('SCHEME NAME : ', scheme_name)
    scheme_path = os.path.join(base_sch_path, scheme_name+'.txt')
    bvals_path = os.path.join(base_sch_path, scheme_name+'.bvals')
    
    dic_name = "MC_simus.pkl"
    dictionary_path = os.path.join(base_dic_path, dic_name)
    with open(dictionary_path, 'rb') as f:
        full_dic = pickle.load(f)
    dic = full_dic[scheme_name]
    
    #%%
    SNRs = [100,50,40,30,20,10]
    for SNR in SNRs:
        if(SNR>0):
            task_name = scheme_name+'_SNR'+str(SNR)
        else:
            task_name = scheme_name+'_SNRINF' 
        
        
        all_idx = np.array([k for k in range(dic['dictionary'].shape[1])])
    
        synth = DG.Synthetizer(scheme_path, 
                            bvals_path, 
                            None, 
                            dictionary_structure = dic,
                            M0_random = False, 
                            task_name = task_name, 
                            num_fasc = num_fasc)
        
        synth.generateStandardSet(num_samples, 
                                  output_path = out_path,
                                  run_id=0, 
                                  SNR_min=SNR, 
                                  SNR_max=SNR, 
                                SNR_dist='uniform', 
                                nu_min=0.15, 
                                crossangle_min=25, 
                                random_seed=None,
                                atoms_idx = None, #To select particular
                                force_overwrite = True)