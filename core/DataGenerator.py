# -*- coding: utf-8 -*-
import json
import pickle
import numpy as np
import os
from tqdm import tqdm
import spherical as sph
import quaternionic as qua
from core import DWI_utils as dwu
from core import sh_utils as shu

#%% Loading data
class DataLoader():
    def __init__(self, base_path, 
                 task_name, 
                 run_id,
                 num_fasc = 2,
                 num_samples = None,
                 sorting = True):
        try:
            path = os.path.join(base_path, "GeneratedData", f"{task_name}")
            filename = f"run-{run_id}_num_fasc-{num_fasc}"
            with open(os.path.join(path, filename + ".pickle"), 'rb') as file:
                self.data = pickle.load(file)
        except FileNotFoundError:
            print(f'File not found for specified path {path}. Triyng without GeneratedData.')
            path = os.path.join(base_path, f"{task_name}")
            filename = f"run-{run_id}_num_fasc-{num_fasc}"
            with open(os.path.join(path, filename + ".pickle"), 'rb') as file:
                self.data = pickle.load(file)
        
        file_num_samples = self.data['DWI'].shape[0]
        if(num_samples is None):
            num_samples = file_num_samples
        if(num_samples != file_num_samples):
            idx_chosen = np.random.choice(file_num_samples, num_samples, replace = False)
            self.idx_chosen = idx_chosen
            for key in self.data.keys():
                if(type(self.data[key]) == np.ndarray):
                    if(self.data[key].shape[0] == file_num_samples):
                        self.data[key] = self.data[key][idx_chosen,...]

#%%
class Synthetizer:
    """Class to generate synthetic data from a given scheme and dictionary."""
    def __init__(self, 
                 scheme_path, 
                 bvals_path, 
                 dictionary_path,
                 dictionary_structure = None, 
                 task_name="default", 
                 M0_random = True, 
                 num_fasc = 2):
        """Initialize the synthetizer with a given scheme and dictionary."""
        assert dictionary_path is None or dictionary_structure is None
        # Load DW-MRI protocol
        bvals = np.loadtxt(bvals_path) # NOT in SI units, in s/mm^2
        scheme = np.loadtxt(scheme_path, skiprows=1)  # only DWI, no b0s

        # Load MF dictionary
        # dictionary_structure = dwu.loadmat(dictionary_path) : legacy dic stored as .mat
        if(dictionary_structure is None):
            with open(dictionary_path, 'rb') as f:
                dictionary_structure = pickle.load(f)

        self.__init_scheme(scheme, bvals)
        self.__init_dictionary(dictionary_structure)
        self.__get_SH_proj() #Getting SH expansion of the canonical signals

        self.num_fasc = num_fasc
        self.task_name = task_name
        
        self.M0_random = M0_random
    
    def __get_SH_proj(self):
        #This is a noiseless case
        D = self.SIM_dict['dictionary']
        ubvals = np.unique(self.SIM_dict['bvals']) #The simulations do not simulate variability of bvals so this is fine
        self.SH_D = [] #List of arrays : not all projections have the same size (we choose the degree in function of the number of available samples at a given shell)
        mask_noB0 = np.abs(self.SIM_dict['bvals']) > 50
        
        
        xyz = self.scheme[mask_noB0,0:3] #b-vectors
        theta,phi,r = shu.Cart2Sph(xyz[:,0], xyz[:,1], xyz[:,2])
        Rs_grid = qua.array.from_spherical_coordinates(theta, phi)
        
        for k,bval in enumerate(ubvals[1:]): #No sh projection for bval == 0
            mask_bval = np.abs(self.SIM_dict['bvals'][mask_noB0] - bval) < 50
            N = int(np.sqrt(np.sum(mask_bval)))
            n_sh = np.array([n for n in range(N+1) for m in range(-n,n+1)])
            mask_H0 = np.array([(n%2 == 0)*(m==0) for n in range(N+1) for m in range(-n,n+1)], dtype = 'bool')
            wig = sph.Wigner(N)
            Ymn = wig.sYlm(0, Rs_grid[mask_bval,:]) 
            _, c_proj_mat = shu.get_SH_proj_mat(N,None, 
                                                n_sh, 
                                                smooth = None, 
                                                cSH = Ymn, 
                                                c_smooth = 1e-10) #Noiseless so very little regularization is needed
            proj = (c_proj_mat @ D[mask_noB0,:][mask_bval,:]).T
            proj[:,~mask_H0] = 0 #Force the signal to be axisymmetric around uz
            self.SH_D.append(proj)
        
        return 1 
    
    
    def __init_dictionary(self, dictionary_structure):
        self.SIM_dict = dictionary_structure
        self.SIM_dict["fasc_propnames"] = [s.strip() for s in dictionary_structure['fasc_propnames']]

        self.interpolator = dwu.init_PGSE_multishell_interp(
            dictionary_structure['dictionary'],
            dictionary_structure['sch_mat'],
            dictionary_structure['orientation'])

    def __init_scheme(self, scheme, bvals):
        ind_b0 = np.where(bvals <= 1e-16)[0]
        ind_b = np.where(bvals > 1e-16)[0]
        num_B0 = ind_b0.size
        sch_mat_b0 = np.zeros((scheme.shape[0] + num_B0, scheme.shape[1]))
        sch_mat_b0[ind_b0, 4:] = scheme[0, 4:]
        sch_mat_b0[ind_b, :] = scheme
        self.scheme = sch_mat_b0
        self.TE = np.mean(self.scheme[:, 6])
        self.num_mris = sch_mat_b0.shape[0]

    def __generateRandomDirections(self, crossangle_min, dir1='fixed', random_seed = None):
        np.random.seed(random_seed)
        crossangle_min_rad = crossangle_min * np.pi / 180
        if dir1 == 'fixed':
            # fixed direction (do not take direction in the Z axis orientation)
            cyldir_1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        elif dir1 == 'random':
            norm1 = -1
            while norm1 <= 0:
                cyldir_1 = np.random.randn(3)
                norm1 = np.linalg.norm(cyldir_1, 2)
            cyldir_1 = cyldir_1 / norm1  # get unit vector
            if cyldir_1[2] < 0:
                # Force half-sphere with positive z
                cyldir_1 = -cyldir_1
        else:
            raise ValueError('dir1 should be either fixed or random')

        # cyldir2 - Enforce min crossing angle
        cyldir_2 = cyldir_1.copy()
        while np.abs(np.dot(cyldir_1, cyldir_2)) > np.cos(crossangle_min_rad):
            norm2 = -1
            while norm2 <= 0:
                cyldir_2 = np.random.randn(3)
                norm2 = np.linalg.norm(cyldir_2, 2)
            cyldir_2 = cyldir_2 / norm2
            if cyldir_2[2] < 0:
                # Force half-sphere with positive z
                cyldir_2 = - cyldir_2
        crossang = np.arccos(np.abs(np.dot(cyldir_1, cyldir_2))) * 180 / np.pi

        return cyldir_1, cyldir_2, crossang

    def generateStandardSet(self, num_samples, output_path = None, run_id=0, SNR_min=20, SNR_max=100, 
                            SNR_dist='uniform', nu_min=0.15, 
                            crossangle_min=30, random_seed=None,
                            atoms_idx = None, force_overwrite = False):
        np.random.seed(random_seed)
        if(output_path is None):
            base_path = os.getcwd()

        
        SNRs = []
        nu_max = 1 - nu_min
        nus = np.zeros((num_samples, self.num_fasc))
        for i in range(num_samples):
            nus_current_sample = np.random.uniform(low = nu_min*self.num_fasc, high = 1, size = self.num_fasc)
            nus_current_sample = nus_current_sample/np.sum(nus_current_sample)
            nus_current_sample = np.flip(np.sort(nus_current_sample))
            nus[i,:] = nus_current_sample[:]
                
            
            if SNR_dist == 'triangular':
                SNR = np.random.triangular(SNR_min, SNR_min, SNR_max, 1)
            elif SNR_dist == 'uniform':
                SNR = np.random.uniform(SNR_min, SNR_max, 1)
            else:
                raise ValueError("Unknown SNR distribution %s" % SNR_dist)
            SNRs.append(SNR)

        data_dic = self.__generator(num_samples, 
                                    nus, 
                                    SNRs, 
                                    crossangle_min, 
                                    random_seed = random_seed, 
                                    atoms_idx = atoms_idx)

        data_dic["parameters"]["type"] = "standard"
        data_dic["parameters"]["run_id"] = run_id
        data_dic['parameters']['num_fasc'] = self.num_fasc
        data_dic['nus'] = nus[:]
        data_dic["parameters"]["SNR_dist"] = SNR_dist
        data_dic["parameters"]["SNR_min"] = SNR_min
        data_dic["parameters"]["SNR_max"] = SNR_max
        data_dic["parameters"]["nu_min"] = nu_min
        data_dic["parameters"]["random_seed"] = random_seed
        self.data_dic = data_dic
        self.save(output_path, force_overwrite)
        
        return 1
    
    def __generator(self, 
                    num_samples, 
                    nus,
                    SNRs, crossangle_min, 
                    random_seed = None, 
                    atoms_idx = None):
        np.random.seed(random_seed)
        assert num_samples == nus.shape[0], "num_samples should be equal to the length of nus1"
        assert num_samples == len(SNRs), "num_samples should be equal to the length of SNRs"

        
        if(atoms_idx is None):
            print('pouet')
            atoms_idx = np.array([k for k in range(self.SIM_dict['dictionary'].shape[1])])
            
        rng = np.random.default_rng(random_seed)
        
        M0 = 1 #Only used if not(M0_random)
        num_coils = 1
        dir1_type = 'random'
        S0_max = np.max(self.SIM_dict["S0_fasc"])
        dot_max = np.cos(crossangle_min * np.pi/180)

        # Prepare output arrays
        IDs = np.zeros((num_samples, self.num_fasc), dtype=np.int32)
        orientations = np.zeros((num_samples, self.num_fasc, 3))
        if(self.num_fasc == 2):
            crossangles = np.zeros((num_samples))
        else:
            crossangles = None
        M0s = np.zeros(num_samples)

        DWI = np.zeros(( num_samples,self.num_mris,), dtype = 'float32')
        DWI_noisy = np.zeros(( num_samples, self.num_mris,), dtype = 'float32')
        DWI_separated = np.zeros((num_samples, self.num_mris,  self.num_fasc), dtype = 'float32')



        print('Generating Voxels...')
        for i in tqdm(range(num_samples)):
            #Random M0 is chosen before iterating on the fascicles because it is needed in each loop
            if(self.M0_random):
                M0 = float(np.random.randint(500, 5000))
                

            DWI_voxel = np.zeros(self.num_mris)
            previous_dirs = np.zeros((self.num_fasc,3))
            for k in range(self.num_fasc):
                #1. Random choice of atom
                ID = rng.choice(atoms_idx, 1, replace = False)
                
                #2. Attribution of the value of nu
                nu = nus[i,k]
                
                #3. Random direction generation : 
                    # TODO : change this by sampling the spherical coordinates 
                    # and excluding the relevant cones at each iteration over the number of fascicles
                fasc_dir = np.zeros((3,1)) #To guarantee at least one direction
                if(k!=0):
                    fasc_dir[:,:] = previous_dirs[k-1,:][:,np.newaxis].copy()
                nb_tried_dirs = 0
                while(nb_tried_dirs < 1 or np.any(np.abs(previous_dirs[:,:]@fasc_dir) > dot_max)): #Not good for a 'big' number of fasc (>3)
                    nb_tried_dirs+=1
                    t = np.array([np.random.uniform(0,2*np.pi)])
                    p = np.array([np.random.uniform(0,np.pi)])
                    fasc_dir[:,:] = shu.Sph2Cart(t,p,1).T
                    
                previous_dirs[k,:] = fasc_dir[:,0]
                #4. Rotation of signal along the chosen orientation
                sig_fasc = dwu.interp_PGSE_from_multishell(self.scheme, 
                                                            ordir=self.SIM_dict['orientation'], 
                                                            newdir=fasc_dir, 
                                                            sig_ms=self.SIM_dict["dictionary"][:, ID], 
                                                            sch_mat_ms=self.SIM_dict["sch_mat"])

                #5. The contribution of this fascicle is added to the noiseless normalized DW MRI signal
                DWI_voxel = DWI_voxel + nu * sig_fasc
                
                #6. The contribution of the fascicle is stored
                DWI_separated[i,:,k] = nu * M0 * sig_fasc
                
                IDs[i, k] = ID
                nus[i, k] = nu
                orientations[i, k, :] = fasc_dir.squeeze()
                
            M0s[i] = M0    
            DWI[i,:] = DWI_voxel #Noiseless signal is stored
            
            #Noise is added and noisy signal is stored
            if(SNRs[i][0]>0):
                sigma_g = S0_max / SNRs[i]
                DWI_voxel_noisy = dwu.gen_SoS_MRI(DWI_voxel, sigma_g, num_coils)
                DWI_noisy[i,:] = M0*DWI_voxel_noisy 
                
        data_dic = {'DWI': DWI,
                    'DWI_noisy': DWI_noisy,
                    'DWI_separated': DWI_separated,
                    'M0s': M0s,
                    'IDs': IDs,
                    'atoms_idx':atoms_idx,
                    'nus': nus,
                    'orientations': orientations,
                    'SNRs': SNRs,
                    'crossangles': crossangles,
                    'parameters': {
                        'task_name': self.task_name,
                        'M0_random': self.M0_random,
                        'dir1_type': dir1_type,
                        'crossangle_min': crossangle_min,
                        'num_samples': num_samples,
                        'num_coils': num_coils,
                        'num_fasc': self.num_fasc,
                        'scheme': self.scheme,
                        'SIM_dict': self.SIM_dict,
                        },
                    }
        
        return data_dic
    
    def save(self, basepath, force_overwrite = False):
        task_name = self.data_dic['parameters']['task_name']
        run_id = self.data_dic['parameters']['run_id']
        num_fasc = self.data_dic['parameters']["num_fasc"] 
        output_path = os.path.join(basepath, "GeneratedData", f"{task_name}")
        
        # Create folder if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        filename = f"run-{run_id}_num_fasc-{num_fasc}"

        if(not(force_overwrite) and os.path.exists(os.path.join(output_path, filename + ".pickle"))):
            raise ValueError('An identical data set already exists. If you want to overwrite it, specify force_overwrite = True') 
        with open(os.path.join(output_path, filename + ".pickle"), 'wb') as handle:
            pickle.dump(self.data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = self.data_dic['parameters'].copy()

        del metadata['SIM_dict']
        del metadata['scheme']

        with open(os.path.join(output_path,filename+'.json'), 'w') as fp:
            json.dump(metadata, fp, indent=4)
