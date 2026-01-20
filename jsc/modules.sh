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
# module load Python/3.11.3

# PyTorch GPU build
module load PyTorch/2.5.1
