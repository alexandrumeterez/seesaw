#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --account=kempner_barak_lab
#SBATCH --output=/n/holyscratch01/barak_lab/Lab/opt-olmo/logs/%j.log
#SBATCH --nodes=1           
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1      
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --partition=kempner_reservation
#SBATCH --constraint=h100

# Custom environment
source ~/.bashrc
mamba deactivate
mamba activate opt-olmo

export CONFIG=configs/kempner/base-c4-t5.yaml+configs/kempner/models/600m.yaml
# export CHECKPOINTS_PATH=/n/holyscratch01/barak_lab/Lab/opt-olmo/ckpts
export CHECKPOINTS_PATH=/n/vast-scratch/kempner_sham_lab/opt-olmo/ckpts

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

# Try playing with max_split_size_mb if you run into OOM errors.
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

if [ -z "${SLURM_NTASKS_PER_NODE+x}" ]; then
  export SLURM_NTASKS_PER_NODE=1
fi

echo "Running with srun"
# Run the script
srun \
  --cpus-per-task=${SLURM_CPUS_PER_TASK} \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    python -u scripts/train.py ${CONFIG} \
      --run_name=olmo_${SLURM_JOB_ID} \
      --save_folder=${CHECKPOINTS_PATH}/${SLURM_JOB_ID}/ \
      ${@}