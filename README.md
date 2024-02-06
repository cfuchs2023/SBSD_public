# SBSD_public
Source code for an implementation of the Sparse Blind Spherical Deconvolution algorithm tailored for diffusion MRI.

The "Core" folder contains all functions and classes to run the SBSD algorithm.
The "scripts" folder contains examples of the use of the functions of the "core" folder. All results presented in the paper can be reproduced using those scripts.

# Scripts
In order to run the scripts, you'll have to modify the variable path_SBSD to the path where you git cloned the repository.
For scripts concerning in vivo data, you have to obtain hcp data from the young adult diffusion dataset (https://www.humanconnectome.org/study/hcp-young-adult), and change the variable base_hcp_path to the path where you stored the data.

# Experiences on synthetic data
Results of Monte Carlo (MC) simulations performed with the open-source MC-DC simulator are stored in MC_simus/MC_simus.pkl. They can be loaded using the pickle package.

The script generate_data_from_MC_simulations.py show how to generate synthetic data with the procedure described in the paper using those precomputed MC simulations. Examples of generated data with a small number of samples are stored in the folder ./data/GeneratedData/01scheme_200_SNR100, which contains two files. The json one stores metadata describing the parameters with which the data was generated, and the .pkl stores the actual data.

Once data has been generated with a sufficient number of samples, you can use the scripts experience_rotation_estimate_on_MC_CSD.py and experience_rotation_estimate_on_MC.py to obtain the results presented in the paper.

# Experiences on in-vivo data
Once you have downloaded the hcp data from the young adult diffusion dataset (https://www.humanconnectome.org/study/hcp-young-adult), you can use the scripts run_SBSD_on_invivo_parallelized.py and run_SH_est_on_invivo_parallelized.py in order to respectively obtain a fiber bundles orientation estimation and SHs expansion estimation. 
In both cases, the code was paralellized using dask (https://www.dask.org/). Therefore, the scheduler and number of processes may need to be adapted to your hardware.



