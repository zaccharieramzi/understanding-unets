#!/bin/bash
#MSUB -r train_unets                # Request name
#MSUB -n 2                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -T 86400                      # Elapsed time limit in seconds
#MSUB -o unet_train_%I.o              # Standard output. %I is the job id
#MSUB -e unet_train_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q normal  # this is just a test script
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
module load flavor/ucx/mt
module load cuda/10.1.105
module load python3/3.7.5
cd $workspace/understanding-unets
pip install --target=$CCCWORKDIR/installed-packages/ --upgrade ./

ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py --ns-train 20 40  &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py --ns-train 30 30  &

wait  # wait for all ccc_mprun(s) to complete.
