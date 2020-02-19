#!/bin/bash
module purge
module load feature/openmpi/net/ib/openib
module load cuda/10.1.105
module load python3/3.7.5

pip install --target=$CCCWORKDIR/installed-packages/ --upgrade ./

export BSD500_DATA_DIR=$CCCSCRATCHDIR/
export BSD68_DATA_DIR=$CCCSCRATCHDIR/
export DIV2K_DATA_DIR=$CCCSCRATCHDIR/
export LOGS_DIR=$CCCSCRATCHDIR/
export CHECKPOINTS_DIR=$CCCSCRATCHDIR/
