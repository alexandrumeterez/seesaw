#!/bin/bash
#SBATCH --job-name=opt_olmo_bsz
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --output=/n/netscratch/kempner_pehlevan_lab/Lab/ameterez/logs/%A_%a.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=150GB
#SBATCH --array=0-4

module load gcc/10.2.0-fasrc01
# Custom environment
source ~/.bashrc
conda deactivate
conda activate opt-olmo

export CONFIG=configs/kempner/base-c4-t5.yaml+configs/kempner/models/150m.yaml
export SWEEP_CONFIG=configs/kempner/sweeps/weight_decay/150m_large_batches_seesaw.yaml


export CHECKPOINTS_PATH=/n/netscratch/kempner_sham_lab/Everyone/ameterez/adam_bs_sched_ckpts

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
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo"
python scripts/kempner/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}