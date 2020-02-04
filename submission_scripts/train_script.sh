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
module load cuda
module load python3/3.7.5
cd $workspace/understanding-unets
pip install --target=$CCCWORKDIR/installed-packages/ ./

ccc_mprun -E '--exclusive' -n 1 ./submission_scripts/unet_20_40.sh  &
ccc_mprun -E '--exclusive' -n 1 ./submission_scripts/unet_30.sh  &

wait  # wait for all ccc_mprun(s) to complete.
