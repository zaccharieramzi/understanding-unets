#!/bin/bash
#MSUB -r test_multi_gpu                # Request name
#MSUB -n 4                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -T 1800                      # Elapsed time limit in seconds
#MSUB -o test_%I.o              # Standard output. %I is the job id
#MSUB -e test_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
module load cuda
module load python3/3.7.5
cd $workspace/understanding-unets

ccc_mprun -E '--exclusive' -n 1 -c 2 -T 1800 -A gch0424 ./submission_scripts/test_train.sh  &
ccc_mprun -E '--exclusive' -n 1 -c 2 -T 1800 -A gch0424 ./submission_scripts/test_train.sh  &

wait  # wait for all ccc_mprun(s) to complete.
