#!/bin/bash
module purge
module load python/3.7.5
module load tensorflow-gpu/py3/2.2.0-dev

export TMPDIR=$SCRATCH/tmp
pip install --no-cache-dir ./
pip install -r learning_wavelets/requirements.txt

export BSD500_DATA_DIR=$SCRATCH/
export BSD68_DATA_DIR=$SCRATCH/
export DIV2K_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
