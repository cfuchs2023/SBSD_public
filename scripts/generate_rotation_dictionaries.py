# -*- coding: utf-8 -*-
import numpy as np
import spherical as sph
import os
import sys 
from tqdm import tqdm
import matplotlib.pyplot as plt

path_SBSD = "E:/github/SBSD_public/" #Put the path where you git cloned SBSD_public
if(not(path_SBSD in sys.path)):
    sys.path.insert(0,path_SBSD)   

from core import sh_utils as shu

#%% Utilities for easy indexation
L = 10 #Dictionaries for even and >0 degrees up to (including) L will be generated
wig = sph.Wigner(L)
Ns = np.array([k for k in range(1,L+1) if k%2==0])

#%%
SAVE = True 
cache_path = os.path.join(path_SBSD, "CachedRotationDictionary")
if(not(os.path.exists(cache_path))):
    os.makedirs(cache_path)

#%% Create dictionary of rotations
idx_mn_in_DN = [wig.Yindex(n,m) for n in Ns for m in range(-n,n+1) ]
DNs = []
full_rots = []
li_xyz_rots = []
nangles = 100
dangle = 0.02

for k,N in tqdm(enumerate(Ns)):
    DN,rots,rtheta,rphi = shu.ComputeRotationDic(N,nangles,dangle = dangle)
    DNs.append(DN)
    xyz_rot = shu.Sph2Cart(rtheta, rphi, 1)
    li_xyz_rots.append(xyz_rot)
    if(SAVE):
        name_D = 'D_nangles_{0}_dangle_{1}_N_{2}.npy'.format(nangles,dangle,N)
        name_xyz = 'xyz_nangles_{0}_dangle_{1}_N_{2}.npy'.format(nangles,dangle,N)
        np.save(os.path.join(cache_path, name_D), DNs[-1])
        np.save(os.path.join(cache_path, name_xyz), li_xyz_rots[-1])


mips = [shu.ComputeMIP(DN) for DN in DNs]
print('MIPs : ', mips)

#%% Parameters for the sanity checks
idx_degree = 2
idx_u = rtheta.shape[0]//4
norms = np.linalg.norm(DNs[idx_degree], axis = 1, keepdims = True)
all_corrs = np.abs(np.conjugate(DNs[idx_degree])@DNs[idx_degree].T)
corrs = all_corrs[idx_u,:]

#%% Sanity check : plot the correlation matrix of a dictionary of a given degree
plt.figure()
plt.title('Correlation matrix for degree {0}'.format(Ns[idx_degree]))
mp = plt.imshow(all_corrs, cmap = 'Greys')
plt.colorbar(mp)
plt.show()

#%% Sanity check : look at one line of the correlation matrix
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection = '3d')
mp = ax.scatter(li_xyz_rots[idx_degree][:,0],
           li_xyz_rots[idx_degree][:,1],
           li_xyz_rots[idx_degree][:,2],
           s = 2,
           marker = 'o',
            c = corrs,
            cmap = 'turbo'
           )
plt.colorbar(mp)
ax.scatter(li_xyz_rots[idx_degree][idx_u,0],
           li_xyz_rots[idx_degree][idx_u,1],
           li_xyz_rots[idx_degree][idx_u,2],
           s = 200,
           marker = 'h',
           color = 'k'
           )

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()

