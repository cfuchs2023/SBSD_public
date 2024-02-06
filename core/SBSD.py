# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import os
from core import OMP as omp
from core import sh_utils as shu
import warnings
import quaternionic as qua
import time
#%%
def LoadRotDic(Ns, dr, nangles, cache_path):
    DNs = []
    li_xyz_rots = []

    for k,N in enumerate(Ns):
        name_D = 'D_nangles_{0}_dangle_{1}_N_{2}.npy'.format(nangles,dr,N)
        name_xyz = 'xyz_nangles_{0}_dr_{1}_N_{2}.npy'.format(nangles,dr,N)
        DN = np.load(os.path.join(cache_path, name_D))
        DNs.append(DN)
        
        xyz_rot = np.load(os.path.join(cache_path, name_xyz))
        li_xyz_rots.append(xyz_rot)

    return DNs, li_xyz_rots

#%%
def run_SBSD(cSH_x, 
             DNs, 
             li_xyz_rots,
             Ns = [6], 
             epsilon = 0.7,
             fixed_numper_of_peaks = False,
             fixed_num_peaks = 2,
             max_n_peaks = 5,
             t_smooth = 1e-6,
             verbosity = 1,
             logpath = None):
    
        
    if(verbosity > 0 and not(fixed_num_peaks is None) and not(epsilon is None)):
        warnings.warn('You have specified a value for a fixed number of peaks (fixed_num_peaks) and one for epsilon. The epsilon will be unused.')
    if(not(fixed_num_peaks is None)):
        assert fixed_num_peaks>0
    if(not(fixed_num_peaks is None)):
        max_n_peaks = fixed_num_peaks
        if(verbosity > 0):
            warnings.warn('You have specified a value for a fixed number of peaks (fixed_num_peaks). The maximum number of peaks is set to this specified value.')
    nsamples = cSH_x.shape[0]
    li_slc_Ns = [slice(n**2,(n+1)**2) for n in Ns]
    
    #Prepare storage of results
    Aks = np.zeros((nsamples, len(Ns),max_n_peaks), dtype = 'uint16') #Indexes of selected U_{u_k}^n
    Wks = np.zeros((nsamples, len(Ns),max_n_peaks), dtype = 'complex') #Associated \nu_k C_{0,k}^n
    rks =  [np.zeros((nsamples,li_slc_Ns[n].stop - li_slc_Ns[n].start), dtype = 'complex') for n in range(len(Ns)) ]  #Residuals
    Cks = np.zeros(((nsamples, len(Ns),max_n_peaks)), dtype = 'complex') #Maximum similarity measures at each iteration
    estimated_s = [np.zeros((nsamples,li_slc_Ns[k].stop - li_slc_Ns[k].start), dtype = 'complex')
                   for k in range(len(Ns)) ] #Reconstructed signals from U_{u_k}^n and \nu_k C_{0,k}^n
    estimated_us = np.zeros((nsamples, len(Ns), max_n_peaks, 3), dtype = 'float64') #Estimated orientations
    num_peaks = np.zeros((nsamples, len(Ns)), dtype = 'int32') #NUmber of peaks found at each voxel
    
    
    #Run the algorithm
    start = time.time()
    for j in tqdm(range(nsamples), disable = verbosity<1):      
        if(not(logpath is None)):
            if((j%(nsamples/20))<1):
                with open(logpath, 'a') as f:
                    f.write('\n Currently processing sample {0} out of {1}. {2} s'.format(j,nsamples, time.time() - start))
        for i,N in enumerate(Ns):
            slc_N = li_slc_Ns[i] #For slicing the coefficients of degree N in the full SH expansion of the signal
            
            #Complex OMP
            Ak, Wk, rk, Ck = omp.complexOMP(DNs[i].T,
                     cSH_x[j,slc_N],
                    N = fixed_num_peaks, 
                    epsilon = epsilon, #Only used if num_peaks is None
                    Tikhonov = True,
                    lamb = t_smooth,
                    max_n_peaks = max_n_peaks)
            
            #Storing results
            num_peak = Ak.shape[0]
            end = min(num_peak, max_n_peaks)
            if(not(num_peak==0)): #When epsilon is in use, it is possible to find no peak
                num_peaks[j,i] = num_peak
    
                estimated_s[i][j,:] = Wk @ DNs[i][Ak,:]
                Wks[j,i,0:end] = Wk[0:end]
                
                Aks[j,i,0:end] = Ak[0:end]
                Wks[j,i,0:end] = Wk[0:end]
                Cks[j,i,0:end] = Ck[0:end]
                    
                estimated_us[j,i,0:end,:] = li_xyz_rots[i][Ak[0:end],:]
                
                rks[i][j,:] = rk[:]
    return Aks, Wks, Cks, estimated_us, estimated_s, rks, num_peaks

#%%
def ComputeCanonicalSignals(max_L, cSH_x, num_peaks, est_us, t_smooth = 1e-6, Ns = None, DNs = None, verbosity = 1, logpath = None):
    #TODO : When the estimation of direction was performed with SBSD, there is no need to recompute the filters since
    #they have been computed in the relevant DN already.
    nsamples = est_us.shape[0]
    max_num_peak = est_us.shape[1]
    weights = np.zeros((nsamples, max_num_peak, max_L//2 + 1), dtype ='complex')
    all_Rus = np.zeros((nsamples,max_num_peak,4))
    start = time.time()
    if(not(logpath is None)):
        with open(logpath, 'a') as f:
            f.write('\n Starting computation')
    
    for j in tqdm(range(nsamples), disable = verbosity <1):
        if(not(logpath is None)):
            if((j%(nsamples/20))<1):
                with open(logpath, 'a') as f:
                    f.write('\n Currently processing sample {0} out of {1}. {2} s'.format(j,nsamples, time.time() - start))
        if(type(num_peaks) == int):
            num_peak = num_peaks
        else:
            num_peak = num_peaks[j]
        
        us = est_us[j,0:num_peak,:]
        s = cSH_x[j,:]
        
        # Computing the rotation associated with u in quaternion representation
        (thet,fi,r) = shu.Cart2Sph(us[:,0], us[:,1], us[:,2])
        Rus = qua.array.from_spherical_coordinates(thet, fi)
        try:
            all_Rus[j,0:num_peak,:] = Rus
        except ValueError:
            print('Num peaks : ', num_peak)
            print('Rus shape', Rus.shape)
            print('All Rus shape', all_Rus.shape)
            raise ValueError('Error')
        #Computing the associated U^n and weights
        for n_idx,n in enumerate(range(0,max_L+1,2)):
            if(type(t_smooth )==float):
                lamb = t_smooth
            else:
                lamb = t_smooth[n_idx]
            slc_n = slice(n**2,(n+1)**2)
            miniD = np.conjugate(shu.MakeSingleOrderRotFilter(n, Rus)).T
            lambI = np.eye(miniD.shape[1], dtype = 'complex') * lamb
            Wns = np.linalg.inv(np.conjugate(miniD).T @ miniD + lambI)@np.conjugate(miniD).T @ s[slc_n]
            weights[j,0:num_peak,n//2] = Wns[:]
    
    return weights, all_Rus
