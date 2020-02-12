#!/bin/bash
#MSUB -r train_learnlets_different_n_samples             # Request name
#MSUB -n 4                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -x
#MSUB -T 86400                      # Elapsed time limit in seconds
#MSUB -o learnlet_train_%I.o              # Standard output. %I is the job id
#MSUB -e learnlet_train_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q normal
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
#module load flavor/ucx/mt
module purge
module load feature/openmpi/net/ib/openib
module load cuda/10.1.105
module load python3/3.7.5
cd $workspace/understanding-unets
pip install --target=$CCCWORKDIR/installed-packages/ --upgrade ./

. ./submission_scripts/env_config.sh

ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/learnlet_training.py -n 200 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/learnlet_training.py -n 100 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/learnlet_training.py -n 50 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/learnlet_training.py -n 10 &

wait  # wait for all ccc_mprun(s) to complete.
