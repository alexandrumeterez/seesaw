#!/bin/bash

export CONFIG=configs/kempner/small-c4-t5.yaml
export CHECKPOINTS_PATH=/n/holyscratch01/barak_lab/Lab/opt-olmo/ckpts

export SLURM_NTASKS_PER_NODE=1

bash scripts/run_with_environment.sh \
    python -u scripts/train.py ${CONFIG} \
      --run_name=olmo_${SLURM_JOB_ID} \
      --save_folder=${CHECKPOINTS_PATH}/${SLURM_JOB_ID}/ \
#       ${@}