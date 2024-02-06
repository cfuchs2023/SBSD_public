# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


#%%
def normalize(x):
    if(len(x.shape) == 2):
        out = x/(np.linalg.norm(x, axis = 1, keepdims = True)+1e-6)
    elif(len(x.shape) == 1):
        out = x/(np.linalg.norm(x, axis = 0, keepdims = True)+1e-6)
    return out

def complexOMP(D, s, N = 2, 
        Tikhonov = True, epsilon = 0.7, lamb = 1e-5, max_n_peaks = 5):
    ''' D : dictionary, contains atoms as columns : dtype is complex
    s : signal : dtype is complex
    N : number of atoms to select 
    epsilon : if N is None, the algorithm stops when all similarity measure bewteen the signal and atoms are below epsilon at a given iteration
    If Tikhonov, lamb is used to regularize the LS problem'''
    #Initialization
    num_atoms = D.shape[1]
    rk = s #residual
    Ak = [] #set of selected atoms
    Ck = [] #values of the maximum correlation at each iteration
    Wk = None #Weights computed at each iteration : will be redefined at each iteration
    mask = np.array([True for i in range(num_atoms)]) #Mask of atoms not yet selected
    # This is used to easily keep track of the indexes of selected atoms
    idx_in_full_dic = np.array([k for k in range(num_atoms)]) #This is used to get the actual idx of the selected atom in the initial (full) dictionary

    stop_criterion = False
    nb_iter = 0 #Number of iteration
    
    if(N is None):
        if(type(epsilon) == float):#If epsilon is a float, the epsilon will be the same for all iter
            epsilon = [epsilon]*int(max_n_peaks)
        elif(len(epsilon)<max_n_peaks):#If there are less epsilon than max_n_peaks, it is padded with the last value given
            epsilon = epsilon + [epsilon[-1]]*(max_n_peaks-len(epsilon))
     
    while(not(stop_criterion)):
        
        #Compute similarity measures
        norm_rk = np.linalg.norm(rk)
        dots = np.abs(np.conjugate(D[:,mask].T)@rk/norm_rk)
        if((N is None) and np.all(dots < epsilon[nb_iter])):
            break
        
        #Select the atom
        raw_selected_atom = np.argmax(dots) #Index of the selected atom in the partial dictionary (atoms not yet selected)
        Ck.append(dots[raw_selected_atom])
        selected_atom = idx_in_full_dic[mask][raw_selected_atom] #IDx of the selected atom in the dictionary with all atoms, including the ones already selected

        if(selected_atom in Ak):
            raise RuntimeError('BUG : Selected atom was already in Ak')
            
        Ak.append(selected_atom) #Store the index of the selected atom
        mask[selected_atom] = False #The same atom cannot be selected twice

        #Recompute all weights associated to the selected atoms
        if(Tikhonov):
            #Tikhonov regularization   
            lambI = np.eye(len(Ak), dtype = 'complex') * lamb
            Wk = np.linalg.inv(np.conjugate(D[:,Ak]).T @ D[:,Ak] + lambI)@np.conjugate(D[:,Ak]).T @ s
        else:
            #No regularization : never use this except for testing
            Wk = (np.linalg.inv(np.conjugate(D[:,Ak]).T @ D[:,Ak])@np.conjugate(D[:,Ak]).T) @ s
        #There is no penalty/constraint for imaginary part of Wks
        #If the input signal is real, the Wks should be real. Experimentally, I have observed on MC data and invivo data
        #that the imaginary part is always near machine zero. If the imaginary part is non zero (>1e-15 or so), there probably is a problem somewhere.
        
        #Compute residual
        rk = rk - D[:,Ak]@Wk
        
        #Check the number of iterations
        nb_iter +=1
        if(not(N is None)):
            stop_criterion = nb_iter == N
        else:
            stop_criterion = nb_iter == max_n_peaks

    return np.array(Ak), np.array(Wk), np.array(rk), np.array(Ck)

def test_complexOMP(num_atoms = 50, atom_size = 400, N = 2 ):
    #Generate dictionary of random atoms : this should be RIP or close to RIP when atom_size -> +inf
    D = np.random.uniform(low = 0, high = 2, size = (atom_size,num_atoms)) + 1j * np.random.uniform(low = 0, high = 2, size = (atom_size,num_atoms))#0.25+np.sin(2*np.pi*fs[np.newaxis,:]*ts[:,np.newaxis])
    D = normalize(D)
    
    #Generate synthetic signal
    gt_Ak = np.random.choice(num_atoms, N, replace = False)
    gt_Ak.sort()
    gt_Wk = np.random.uniform(low = 0.25, high = 1, size = N) + 1j * np.random.uniform(low = 0.25, high = 1, size = N)
    s = D[:,gt_Ak]@gt_Wk
    
    print('Ground truth atoms : ', gt_Ak)
    print('Ground truth weights : ', gt_Wk)
    
    #Solve with OMP
    Ak, Wk, rk, Ck = complexOMP(D,
             s,
            N = N)
    #Check results
    print('Atoms found : ', Ak)
    print('Weights found : ', Wk)
    print('Ck : ', Ck)
    
    print('MSE : ', np.mean((s - D[:,Ak]@Wk)**2))
    
    fig,axs = plt.subplots(1,2, squeeze = False)
    axs[0,0].plot(np.abs(s), '-hk', markersize = 6, label = 'Ground truth')
    axs[0,0].plot(np.abs(D[:,Ak]@Wk), '--*c', markersize = 3, label = 'Estimated')
    plt.legend()
    
    axs[0,1].plot(np.abs(s - D[:,Ak]@Wk), '--g*', label = 'Reconstructed Residual', markersize = 1.5)
    plt.legend()
    plt.show()
    
    return None

def MajorityVote(X, W = None):
    #X : matrix on which to perform majority vote ; the last axis contains the votes of each individual
    #W : weights of the votes for each individual ; should be of shape (X.shape[2],)
    
    if(not(W is None)):
        assert W.shape[0] == X.shape[2]
        W_repeated = np.repeat(W,X.shape[2])
    mat = np.zeros((*X.shape[0:2], X.shape[2]**2), dtype = 'float32')
    idx_start = 0
    for j in range(X.shape[2]):
        idx_end = idx_start + X.shape[2]
        #TODO : replace loop with mat = np.repeat(X, reps = (1,1,X.shape[2])) == np.tile(...)
        mat[:,:,idx_start:idx_end] = X == np.tile(X[:,:,j:j+1], reps = (X.shape[2]))
        idx_start = idx_end
        
    if(not(W is None)):
        mat = mat * W_repeated[np.newaxis,np.newaxis,:]
        
    mat = np.cumsum(mat, axis = 2)
    
    idx_retrieve_votes = np.array([X.shape[2] * (k+1) - 1 for k in range(X.shape[2])])
    votes = np.diff(mat[:,:,idx_retrieve_votes], prepend = 0, axis = 2)
    idx_choices = np.argmax(votes, axis = 2)
    
    out = np.take_along_axis(X, idx_choices[:,:,np.newaxis], axis = 2).squeeze()
    return out, votes
        
#%%
if(__name__ == '__main__'):
    useless = test_complexOMP(N=4)
