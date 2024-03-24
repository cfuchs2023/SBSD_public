# -*- coding: utf-8 -*-
#%% Load module
import numpy as np
from tqdm import tqdm
import quaternionic as qua    
import spherical as sph
import matplotlib.pyplot as plt


#%%
#/!\ we use the convention phi = longitude (u_x,OM) and theta = colatitude (u_z,OM) 
# i.e. the same as dipy
#north pole is nu and theta = 0, phi = 0
#scipy uses the opposite : phis is colatitude and theta is longitude :'(

def Cart2Sph(x,y,z):
    '''
    Theta is the colatitude
    Phi is the longitude
    (dipy convention)
    '''
    
    xysq = x**2 + y**2
    r = np.sqrt(xysq + z**2)
    xysq = xysq/r**2
    
    thetas = -np.arctan2(z/r, np.sqrt(xysq)) + np.pi/2
    phis = np.arctan2(y/r,x/r)
    
    return thetas, phis, r

def Sph2Cart(thetas,phis, r):
    ''' thetas is a vector storing values in the following order : 
        theta0,theta1,...,thetan, theta0, theta1, ... thetan.....
        phis is a vector storing values in the following order
        phi0,phi0,phi0,phi0,,....,phim,phim,phim,phim
        dipy convention (theta is colatitude, phi is longitude)
        '''
    shape = thetas.shape
    xyz = np.zeros((*shape,3))
    xyz[...,0] = r * np.cos(phis) * np.sin(thetas)
    xyz[...,1] = r * np.sin(phis) * np.sin(thetas)
    xyz[...,2] = r * np.cos(thetas)
    return xyz

#%% Some utilities for indexing of SH harmonics
def mn_to_lin(m,n): #for easy acces to Y_m^n in Ymn : Y_m^n = Ymn[:,mn_to_lin(m,n)] 
    #if the spherical harmonics are in order n,m = (0,0), (1,-1), (1,0), (1,1) etc... in Ymn
    if(m>n or m<-n):
        raise ValueError('m has to be in [|-n,n|] but got m = {0} and n = {1}'.format(m,n))
    lin_idx = (n-1) * (n+1) + (m+n) + 1
    return lin_idx

def slice_to_n(n):
    return slice(0,n*(n+2)+1)

def lin_to_mn(idx): #Ymn[:,idx] = Y^{lin_to_mn(idx)[0]}_{lin_to_mn(idx)[1]}
    n = 0
    m = 0
    #This one is dirty and you should probably use a Wigner object from spherical
    try:
        candidate_lin = mn_to_lin(m,n)
    except ValueError:
        candidate_lin = -1
    while(candidate_lin!=idx):
        n+=1
        m = idx-(n+1)*(n-1)-n-1
        
        try:
            candidate_lin = mn_to_lin(m,n)
        except ValueError:
            candidate_lin = -1
    return m,n


#%%
def get_SH_proj_mat(L, SH, n_sh, smooth = 2e-3, cSH = None, c_smooth = 1e-6):
    mask = np.array([n%2 == 0 for n in range(L+1) for m in range(-n,n+1)], dtype = 'bool')
    #Regularization proposed by Descoteaux et al, see 10.1002/mrm.21277
    L = -n_sh[mask] * (n_sh[mask] + 1)  # Laplace Beltrami operator for each SH basis function, 1-D array
    L = np.diag(L)
    proj_mat = None
    c_proj_mat = None
    if(not(cSH) is None):
        c_proj_mat = np.zeros(cSH.T.shape, dtype = 'complex')
        c_proj_mat[mask,:] = np.linalg.inv(cSH[:,mask].T @ cSH[:,mask] + c_smooth * L**2) @ cSH[:,mask].T
    if(not(SH) is None):
        proj_mat = np.zeros(SH.T.shape, dtype = 'complex')
        proj_mat[mask,:] = (np.linalg.inv(SH[:,mask].T @ SH[:,mask] + smooth * L**2) @ SH[:,mask].T).astype('float32')
    return proj_mat, c_proj_mat

#%% To get the Tournier basis from spherical wigner object

def get_Tournier_basis(L, theta_shell, phi_shell):
    '''This function is untested and could contain errors. Good luck. '''
    wig = sph.Wigner(L)
    Rs = qua.array.from_spherical_coordinates(theta_shell, phi_shell)
    Ymn = wig.sYlm(0, Rs) 
    size_tournier_basis = np.sum([(2*l+1) for l in range(L+1) if(l%2==0)])
    tournier_Ymn = np.zeros((Rs.shape[0],size_tournier_basis, ), dtype = 'complex')
    
    idx = 0
    for l in range(L+1):
        for m in range(-l,l+1):
            if(l%2 == 0):
                
                if(m<0):
                    tournier_Ymn[:,idx] = np.sqrt(2)* Ymn[:,wig.Yindex(l,-m)].imag
                elif(m>0):
                    tournier_Ymn[:,idx] = np.sqrt(2)* Ymn[:,wig.Yindex(l,m)].real
                else:
                    tournier_Ymn[:,idx] = Ymn[:,wig.Yindex(l,m)]
                idx+=1
    #Sanity check
    if(np.any(tournier_Ymn.imag>1e-12)):
        raise RuntimeError('Non zero imaginary values found in tournier_Ymn. Something went wrong, sorry.')
    return tournier_Ymn.real

#%%
def MakeRotFilter(l_max, rots):
    #This filter can perform rotation of axisymmetric (AROUND u_z) signal
    #It does not work if the signal is not axisymmetric around u_z
    #rots is a quaternionic array
    wigner = sph.Wigner(l_max)
    n_rots = rots.shape[0]
    rot_filter_coeffs = np.zeros((l_max+1)**2, dtype = 'complex')
    idx_start = 0
    for l in range(l_max+1):
        idx_end = idx_start + 2*l+1
        rot_filter_coeffs[idx_start:idx_end] = np.sqrt(4*np.pi/(2*l+1))
        idx_start = idx_end
    rot_filter_coeffs = sph.Modes(rot_filter_coeffs,0)
    
    Id = np.eye((l_max+1)**2, dtype = 'complex')
    rotation_filters = np.zeros((n_rots, (l_max+1)**2), dtype = 'complex')
    for l in tqdm(range(l_max+1)):
        for m in range(-l,l+1):   
            lm_idx = wigner.Yindex(l,m)
            rotation_filters[:,lm_idx] = rot_filter_coeffs[lm_idx] * wigner.evaluate(sph.Modes(Id[:,lm_idx],0),rots.conjugate())
    
    return rotation_filters


def MakeSingleOrderRotFilter(l, rots):
    #This filter can perform rotation of axisymmetric (AROUND u_z) signal
    #It does not work if the signal is not axisymmetric around u_z
    #rots is a quaternionic array
    #Compared to the legacy, this one is adapted to the Healy convolution instead of the sifting convolution
    wigner = sph.Wigner(l)
    n_rots = rots.shape[0]
    rot_filter_coeff = np.sqrt(4*np.pi/(2*l+1))
    
    rotation_filters = np.zeros((n_rots, 2*l+1), dtype = 'complex')
    Id = np.eye((l+1)**2, dtype = 'complex')
    for m in range(-l,l+1):
        lm_idx = wigner.Yindex(l,m)
        rotation_filters[:,m+l] = rot_filter_coeff * wigner.evaluate(sph.Modes(Id[:,lm_idx],0),rots)
    
    return rotation_filters

def RotateAxisymSignal(SH_x, rot):
    #Complex coefficients only!!! Does not work with real spherical harmonics
    #rot is a quaternionic array
    #SH_x is the exapnsion of a REAL AXISYMMETRIC (around u_z) SIGNAL
    #Does not work for asisymmetric but not around u_z !!!!!!!
    l_max = int(np.sqrt(SH_x.shape[1])-1)
    wigner = sph.Wigner(l_max)
    assert SH_x.dtype in [np.dtype('complex'),np.dtype('complex64')]
    SH_rot_x = np.zeros(SH_x.shape, dtype = 'complex')
    rotation_filters = MakeRotFilter(l_max, rot)
    idx_start = 0
    for l in range(l_max+1): 
        idx_end = idx_start + 2*l+1
        SH_rot_x[:,idx_start:idx_end] = SH_x[:,wigner.Yindex(l,0)].ndarray[:,np.newaxis] *  np.conjugate(rotation_filters[0,idx_start:idx_end])
        idx_start = idx_end
    return SH_rot_x

#%%
def __compute_rotation_grid(ntheta,dangle):
    
    #First we sample the hemi-sphere ; we will convert to rotations later on
    #Each couple (theta,phi) is the location at which u_z ends when the associated rotation u is applied to u_z
    lin_theta = np.linspace(0,np.pi,ntheta, endpoint = False)
    li_phi = [np.array([0], dtype = 'float64')]  #We don't want to have multiple theta = 0
    
    for k in range(1,ntheta):#We don't want to have multiple theta = 0
        #For each iso theta ring, we sample it with values of phis such that the separation angle between two points of the grid is dangle
        theta = lin_theta[k]
        num_phis = int(np.pi/dangle)+1
        li_phi.append(np.linspace(0,np.pi,num_phis, endpoint = False))

    number_of_phis = [x.shape[0] for x in li_phi]

    theta = np.repeat(lin_theta, number_of_phis)
    phi = np.zeros([theta.shape[0]])
    start = 0
    for k in range(ntheta):
        end = start + li_phi[k].shape[0]
        phi[start:end] = li_phi[k][:]
        start = end
    Ru = qua.array.from_euler_angles(phi, theta, np.zeros(theta.shape))
    return theta,phi,Ru

def ComputeRotationDic(l,nbeta,dangle = 0.1):
    #Pay attention to conventions : first rotation is gamma about uz, second beta bout uy, then alpha about uz (eg zyz)
    #Therefore, alpha is phi and beta is theta for axisymmetric signals
    #Conventions can be a headache sometimes !!!
    theta,phi,Ru = __compute_rotation_grid(nbeta, dangle) 
    # rots = qua.array.from_euler_angles( beta, alpha,np.zeros(beta.shape),) 
    # rots = rots.inverse
    D = np.conjugate(MakeSingleOrderRotFilter(l, Ru))
    return D,Ru,theta,phi

def ComputeMIP(D):
    MIP = 0
    mat = (np.conjugate(D) @ D.T).real
    MIP = np.max(np.triu(mat,k=1))
    return MIP





#%%
def match_peaks(gt_us, est_us):
    matched_est_us = np.zeros(est_us.shape, est_us.dtype)
    num_fasc = gt_us.shape[1]
    num_samples = gt_us.shape[0]
    
    # For easy indexing
    line_mat = np.array([[i for j in range(num_fasc)] for i in range(num_fasc)])
    column_mat = np.array([[j for j in range(num_fasc)] for i in range(num_fasc)])
    
    for j in range(num_samples):
        all_dots = np.abs(gt_us[j,:,:] @ est_us[j,:,:].T) #all_dots[a,b] = < gt_a, est_b >
        mask_selected = np.zeros((num_fasc, num_fasc), dtype = 'bool') 
        
        if(False): #Some checks if one wants to
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1,projection='3d')
            for a in range(num_fasc):
                ax.quiver(0,0,0,
                          gt_us[j,a,0], gt_us[j,a,1],gt_us[j,a,2],
                          color = 'r')
            for a in range(num_fasc):
                ax.quiver(0,0,0,
                          est_us[j,a,0], est_us[j,a,1],est_us[j,a,2],
                          color = 'b')
            ax.axes.set_xlim3d(-1,1)
            ax.axes.set_ylim3d(-1,1)
            ax.axes.set_zlim3d(-1,1)
            plt.show()
        
        permut_gt = []
        permut_est = []
        
        for k in range(num_fasc):
            lin_idx = np.argmax(all_dots[~mask_selected])
            id_gt = line_mat[~mask_selected][lin_idx]
            id_est = column_mat[~mask_selected][lin_idx]
            
            permut_gt.append(id_gt)
            permut_est.append(id_est)
            
            mask_selected[id_gt,:] = True
            mask_selected[:,id_est] = True
        
        permut_est = np.array(permut_est)
        permut_gt = np.array(permut_gt)
        idx_sort_permut_according_to_gt = np.argsort(permut_gt)
        matched_est_us[j,:,:] = est_us[j,permut_est[idx_sort_permut_according_to_gt],:]
        
    return matched_est_us

#%%
def Compute_q(u, SH_x, N):
    ''' Compute the correlation between the filter associated with u and the SH expansion of x
    It should be one if SH_x is an axisymmetric signal of orientation u'''
    
    #Compute quaternion representation of the rotation associated with direction u
    thet,fi = Cart2Sph(u[0:1],u[1:2],u[2:])
    ortho_u = np.array([u[1],-u[0],0])
    ortho_u = ortho_u/np.linalg.norm(ortho_u)
    Ru = qua.array.from_axis_angle(thet * ortho_u)
    
    slc_N = slice(N**2, (N+1)**2)
    filt = np.conjugate(MakeSingleOrderRotFilter(N,Ru[np.newaxis,:]))
    q = np.abs(np.conjugate(filt[0,:]) @ SH_x[slc_N])/np.linalg.norm(SH_x[slc_N])
    
    return q, Ru

def CorrRotfilterWithD(u, D, N):
    '''Compute the correlation of the filter associated with u with all the filters contained in D'''
    #Compute quaternion representation of the rotation associated with direction u
    thet,fi = Cart2Sph(u[0:1],u[1:2],u[2:])
    ortho_u = np.array([u[1],-u[0],0])
    ortho_u = ortho_u/np.linalg.norm(ortho_u)
    Ru = qua.array.from_axis_angle(thet * ortho_u)
    
    #slc_N = slice(N**2, (N+1)**2)
    filt = np.conjugate(MakeSingleOrderRotFilter(N,Ru[np.newaxis,:]))
    qs = np.abs(np.conjugate(D) @ filt[0,:])
    
    return qs, Ru

def CorrSHWithD(SH_x, D, N):
    '''Compute the correlation of SH_x at degree N with filters of D'''
    slc_N = slice(N**2, (N+1)**2)
    qs = np.abs(np.conjugate(D) @ SH_x[slc_N])/np.linalg.norm(SH_x[slc_N])
    return qs

#%%
def build_mask(dic):
    bvals = dic['bvals']*1e-6
    ubvals = np.unique(bvals)
    num_dir = (bvals.shape[0]-1)//(ubvals.shape[0]-1)
    bvecs = dic['sch_mat'][:,0:3]
    mask_nob0 = bvals>0
    x = bvecs[mask_nob0,2]
    mask_bvals = np.zeros((ubvals.shape[0]-1,np.sum(mask_nob0)), dtype = 'bool')
    all_idx_sort = np.zeros((ubvals.shape[0]-1,num_dir), dtype = 'int64')
    for j,bval in enumerate(ubvals[1:]):
        mask_bvals[j,:] = np.abs(bvals[mask_nob0]-bval)<30
        all_idx_sort[j,:] = np.argsort(x[mask_bvals[j,:]])
    return mask_bvals,ubvals,mask_nob0,all_idx_sort

def build_Rs(dic):
    bvecs = dic['sch_mat'][:,0:3]
    mask_nob0 = dic['bvals']>0
    xyz = bvecs[mask_nob0,:]
    theta,phi,r = Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
    grid = np.zeros((theta.shape[0],2))
    grid[:,0] = theta[:]
    grid[:,1] = phi[:]
    Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
    return Rs_grid,theta,phi,r,xyz


#%%
def pp_impulses(s_sig, z, ws = 12, start_z = 0.9):
    #Performs moving average on the ends of the signal to remove some ringings
    
    # z \in [0.9 , 1]
    mask = z>start_z
    sig_ma = np.zeros(s_sig.shape)
    sig_ma[:,:,~mask] = s_sig[:,:,~mask]
    
    nsamples, nfasc, ndirs = s_sig.shape
    trail_means = np.mean(s_sig[:,:,mask], axis = -1, keepdims = True)
    padding = trail_means * np.ones((nsamples, nfasc, ws), dtype = 'float32')
    padded_sig = np.zeros((nsamples, nfasc, ndirs+ws), dtype = 'float32')
    padded_sig[:,:,0:ndirs] = s_sig[...]
    padded_sig[:,:,ndirs:] = padding[...]
    padded_mask = np.concatenate([mask, np.ones(ws, dtype = 'bool')])
    
    cs_sig = np.cumsum(padded_sig, axis = -1)

    temp = np.zeros((nsamples,nfasc, np.sum(mask)), dtype = 'float32')
    temp = (cs_sig[:,:,padded_mask][:,:,ws:] - cs_sig[:,:,padded_mask][:,:,:-ws])/(2*ws)
    sig_ma[:,:,mask] = (temp[:] - temp[:,:,0:1]) + s_sig[:,:,mask][:,:,0:1]
    
    # Since the signal is axisymmetric around uz we just copy the flipped last part to the beginning 
    mask_beginning = np.flip(mask)#z<-start_z
    sig_ma[:,:,mask_beginning] = np.flip(sig_ma[:,:,mask], axis = -1)

    
    return sig_ma