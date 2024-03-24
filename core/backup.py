# -*- coding: utf-8 -*-
#%% 
def ComputeCanonicalSignals3(max_L, 
                             cSH_x, #one column of cSH_x corresponds to \Sigma in the paper
                             num_peaks,
                             est_us,
                             t_smooth = 1e-6, 
                             Ns = None, 
                             DNs = None, 
                             verbosity = 1, 
                             logpath = None,
                             groundtruth = None):
    import matplotlib.pyplot as plt
    #With regularization on the derivative
    #TODO : When the estimation of direction was performed with SBSD, there is no need to recompute the filters since
    #they have been computed in the relevant DN already.
    state = {}
    nsamples = est_us.shape[0]
    max_num_peak = est_us.shape[1]
    num_coeffs = max_L//2 + 1 #Number of SH of even degree n <= max_L and order 0
    
    
    
    weights = np.zeros((nsamples, max_num_peak, num_coeffs), dtype ='complex')
    start = time.time()
    mask_even_n = np.array([n%2==0 for n in range(0,max_L+1) for m in range(-n,n+1)], dtype = 'bool')
    
    #Now we build a grid and a matrix of spherical harmonics evaluated on a simple grid
    # We only need to evaluate on a line from z=0 to z=1 on the sphere because we will only use SH with m = 0
    
    #DONE : we only need the SH with m = 0 which are proportional to legendre polynomials
    #use scipy.special.legendre
    #Build the line
    M = 100
    z = np.linspace(0,1,M)
    diff_z = z[1:] - z[0:-1]
    inv_diff_z = 1/diff_z
    
    # xyz = np.zeros((M, 3), dtype = 'float32')
    # xyz[:,2] = z[:]
    # xyz[:,0:2] = np.random.uniform(0,1,size = (M,2))
    # xyz[:,0:2] = xyz[:,0:2]/(np.linalg.norm(xyz[:,0:2], axis = -1, keepdims = True))
    # xyz[:,0:2] = xyz[:,0:2] * np.sqrt(1-z[:,np.newaxis]**2)
    
    # #Convert to quaternions
    # theta,phi,r = shu.Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
    # Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
    
    # #Evaluate the SHs
    # wig = sph.Wigner(max_L)
    # Ymn = wig.sYlm(0, Rs_grid)
    
    # z = xyz[:,2]
    # diff_z = z[1:] - z[0:-1]
    # inv_diff_z = 1/diff_z
    
    # mask_H0_even_n = np.array([m==0 and n%2 == 0 for n in range(0,max_L+1) for m in range(-n,n+1)], dtype = 'bool')
    # sorted_Y0n =  Ymn[:,mask_H0_even_n] #
    
    
    Y0n = np.zeros((M,max_L//2+1), dtype = 'complex')
    for k,n in enumerate(range(0,max_L+1,2)):
        Y0n[:,k] =  np.sqrt((2*n+1)/(4*np.pi)) * eval_legendre(n,z) #So evaluating all SH was useless : use scipy.special.legendre instead
    
    # Initialize log files
    if(not(logpath is None)):
        with open(logpath, 'a') as f:
            f.write('\n Starting computation')
    
    # Regularizations
    regul_derivative = False
    regul_nng = True
    max_iter_regul = 80
    if(regul_derivative):
        state['f_val'] = np.zeros((cSH_x.shape[0], max_iter_regul), dtype = 'float32')
        state['num_iter'] = np.zeros(cSH_x.shape[0])
    if(regul_nng):
        state['num_iter'] = np.zeros(cSH_x.shape[0])
        state['f_val'] = np.zeros((cSH_x.shape[0], max_iter_regul), dtype = 'float32')
    # Start looping on the voxels
    for j in tqdm(range(nsamples), disable = verbosity <1):
        if(not(logpath is None)):
            if((j%(nsamples/20))<1):
                with open(logpath, 'a') as f:
                    f.write('\n Currently processing sample {0} out of {1}. {2} s'.format(j,nsamples, time.time() - start))
        if(type(num_peaks) == int):
            num_peak = num_peaks
        else:
            num_peak = num_peaks[j] #num_peak is K in the paper
        
        us = est_us[j,0:num_peak,:]
        s = cSH_x[j,:]
        
        #TODO : cleanup the code
        
        # Computing the rotation associated with u in quaternion representation
        (thet,fi,r) = shu.Cart2Sph(us[:,0], us[:,1], us[:,2])
        Rus = qua.array.from_spherical_coordinates(thet, fi)
             
        #Prepare A
        num_rows_A = np.sum([2*n+1 for n in range(0,max_L+1,2)])
        A = np.zeros((num_rows_A, num_peak*(max_L//2+1)), dtype = 'complex') 
        row_start_A = 0
        for n_idx,n in enumerate(range(0,max_L+1,2)):
            if(type(t_smooth )==float):
                lamb = t_smooth
            else:
                lamb = t_smooth[n_idx]
            row_end_A = row_start_A + 2*n +1
            miniD = np.conjugate(shu.MakeSingleOrderRotFilter(n, Rus)).T
            A[row_start_A:row_end_A, num_peak*n_idx:num_peak*(n_idx+1)] = miniD[:]
            row_start_A = row_end_A
            
        #Prepare Z for regularization on f'(cos(theta))
        Z = np.zeros((M*num_peak, num_peak * num_coeffs ), dtype = 'complex')
        for k in range(num_peak):
            idx_cols = np.array([j*num_peak+k for j in range(0,max_L//2+1)])
            try:
                Z[M*k:M*(k+1), idx_cols] = Y0n[:,:]
            except IndexError:
                print('prout')
                
        
        #Prepare D and \tilde{D} for regularization on f'(cos(theta))
        #This is not a matrix of ones because the theta are not uniformly distributed
        D = None
        D_tilde = np.zeros((M*num_peak, M*num_peak ), dtype = 'float64')

        for k in range(num_peak):
            if(D is None): #We only need to compute D once
                D = np.zeros((M,M), dtype = 'float64')
                np.fill_diagonal(D[0:-1,0:-1], -inv_diff_z)
                np.fill_diagonal(D[:-1,1:], inv_diff_z)
                D[-1,-2] = -inv_diff_z[-1] 
                D[-1,-1] = inv_diff_z[-1] 
                
            D_tilde[k*M:(k+1)*M, k*M:(k+1)*M,] = D[:,:]
                
        D_tildeZ = D_tilde@Z
        if(np.any(np.isnan(D_tildeZ))):
            print('\n ***** Nan detected in D_tildeZ.')
        
        # Intializing the algorithm with the non regularized least square
        lambda_n = np.array([1 for n in range(0,max_L+1,2)]) * t_smooth #np.array([(n*(n+1))**2 for n in range(0,max_L+1,2)]) * t_smooth
        lambI = np.eye(A.shape[1])*np.diag(np.tile(lambda_n, num_peak))
        X0 = np.linalg.inv(np.conjugate(A).T @ A )@np.conjugate(A).T @ s[mask_even_n]
        Xk = X0.copy()
        

        #Iteratively refining the solution with the prior on the derivative       
        
        if(regul_derivative):
            mu = 1e-6
            eps = 1e-4
            tau = 1e-1
            max_iter = 40
            nb_iter = 0
            all_neg = np.all((D_tildeZ@X0).real<0) #If the initial solution is feasible there is nothing to do
            if(not(all_neg)):
                Xk = Xk+2*eps
                D_tildeZX = (D_tildeZ@X0).real
                Wk = np.diag(D_tildeZX.squeeze()>0)
                Wk_nng = np.diag((Z@X0).squeeze()>0)
                var_x = (np.linalg.norm(Xk-X0)/np.linalg.norm(X0))
                stop_criterion = var_x<eps
                while(not(stop_criterion) and nb_iter<=max_iter_regul-1):
                    inv_mat = np.linalg.inv(A.T @ A + lambI + mu*(Wk@D_tildeZ).T @ (Wk@D_tildeZ))
                    Xk = inv_mat @ A.T @ s[mask_even_n]
                    D_tildeZX = (D_tildeZ@Xk).real
                    f_val = np.linalg.norm(D_tildeZX)
                    # plt.figure()
                    # plt.title(f'Sample : {j} | Iter : {nb_iter} | f val : {f_val}')
                    # plt.plot(D_tildeZX)
                    # plt.show()
                    
                    state['f_val'][j,nb_iter] = f_val
                    Wk = np.diag(D_tildeZX.squeeze()>0)
                    var_x = (np.linalg.norm(Xk-X0)/np.linalg.norm(X0))
                    stop_criterion = var_x<eps
                    X0 = Xk.copy()
                    mu = mu*1.5
                    nb_iter+=1
                    if(f_val<tau):
                        break
                state['num_iter'][j] = nb_iter
                
        if(regul_nng): 
            mu_start = 1e-4
            eps = 1e-6
            max_iter = 80
            nb_iter = 0
            all_pos = np.all((Z@X0).real>-1e-2) #If the initial solution is feasible there is nothing to do
            # print('\n\n ========= SAMPLE : '+str(j))
            if(not(all_pos)):
                                    
                mu = mu_start
                Xk = Xk+2*eps
                ZX0 = (Z@X0).real
                ZX = ZX0.copy()
                W0_nng = np.diag((Z@X0).squeeze()<-1e-2)
                Wk_nng = W0_nng.copy()
                var_x = (np.linalg.norm(Xk-X0)/np.linalg.norm(X0))
                while(not(var_x<eps) and nb_iter<=max_iter_regul-1):
                    # plt.figure()
                    # plt.title(f'NNG Sample : {j} | Iter : {nb_iter} | |Wk_nngZX| {np.linalg.norm(Wk_nng@ZX)}')
                    # plt.plot(ZX)
                    # plt.show()
                    #ZX0_norm = np.linalg.norm(ZX)
                    W0_nng = Wk_nng.copy()
                    Wk_nng = np.diag(ZX.squeeze()<-1e-2)
                    inv_mat = np.linalg.inv(A.T @ A + lambI + mu*(Wk_nng@Z).T @ (Wk_nng@Z))
                    update_mat = inv_mat @ A.T
                    Xk =  update_mat @ s[mask_even_n]
                    ZX = (Z@Xk).real   
                    f_val = np.linalg.norm(ZX)
                    state['f_val'][j,nb_iter] = f_val
                    
                    var_x = (np.linalg.norm(Xk-X0)/np.linalg.norm(X0))
                    ZX0 = ZX.copy()
                    X0 = Xk.copy()
                    mu = mu*1.1
                    nb_iter+=1
                state['num_iter'][j] = nb_iter
               
        
        
        
        weights[j,0:num_peak,:] = Xk.reshape(( weights.shape[-1],num_peak,)).T # "It just works." - Todd Howard 
        #The weights in x are ordered as such : first all coefficients of degree 0 for signal 1, up to K, then all coefficients of degree 2 for signal 1 up to K, etc..
    return weights, state  #We return all computed quaternions for sanity checks purposes   
    
    
    
    
    
    
    
    
    
    
    
    