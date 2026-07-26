#!/bin/bash
#SBATCH --job-name=opt-olmo
#SBATCH --account=kempner_sham_lab
#SBATCH --output=/n/holyscratch01/sham_lab/opt-olmo/logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB		
#SBATCH --partition=kempner_reservation
#SBATCH --constraint=h100
#SBATCH --array=1-7
#SBATCH --exclude=holygpu8a15401

# Custom environment
source ~/.bashrc
conda deactivate
conda activate opt-olmo

export CONFIG=configs/kempner/base-c4-t5.yaml+configs/kempner/models/150m.yaml
export SWEEP_CONFIG=configs/kempner/sweeps/momentum_llama_lite.yaml
export CHECKPOINTS_PATH=/n/holyscratch01/sham_lab/opt-olmo/ckpts
# export CHECKPOINTS_PATH=/n/holyscratch01/barak_lab/Lab/opt-olmo/ckpts
# export CHECKPOINTS_PATH=/n/vast-scratch/kempner_sham_lab/opt-olmo/ckpts



# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

# Try playing with max_split_size_mb if you run into OOM errors.
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/kempner/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}