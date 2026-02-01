# ### ALT ###

# module purge
# module load Stages/2025
# module load GCC OpenMPI
# # Some base modules commonly used in AI
# module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
# module load Flask Seaborn

# # ML Frameworks
# module load  PyTorch scikit-learn torchvision PyTorch-Lightning
# module load tensorboard
# module load h5py

### NEU ###

module purge

module load Stages/2025
module load GCCcore/.13.3.0
# module load GCCcore/.12.3.0  # for PyTorch/2.1.2 (gibt es nicht mehr in module)
module load Python/3.12.3

# PyTorch GPU build
module load PyTorch/2.5.1
# module load PyTorch/2.1.2 # for mmcv 2.2.0 (error ops)

# ### NEU 2 ###

# module purge

# module load Stages/2023
# module load Python/3.10.4
# python -V

# ### NEU 3 ###

# module purge --force
# module load Stages/2023
# module load GCC/11.3.0
# module load OpenMPI/4.1.4

# module spider Python
# module spider Python/3.9

# module load Python/3.9.13
# module load PyTorch/1.12.0-CUDA-11.7
