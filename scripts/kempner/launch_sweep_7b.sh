#!/bin/bash
#SBATCH --job-name=opt-olmo
#SBATCH --account=kempner_sham_lab
#SBATCH --output=/n/holyscratch01/sham_lab/opt-olmo/logs2/%A_%a.log
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=1000GB		
#SBATCH --partition=kempner_h100
#SBATCH --array=1
#SBATCH --exclude=holygpu8a15401

# Custom environment
source ~/.bashrc
conda deactivate
conda activate opt-olmo

export CONFIG=configs/kempner/base-c4-t5.yaml+configs/kempner/models/7b.yaml
export SWEEP_CONFIG=configs/kempner/sweeps/optimizers_benchmark.yaml
export CHECKPOINTS_PATH=/n/holyscratch01/sham_lab/opt-olmo/ckpts


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