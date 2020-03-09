#!/bin/bash
#SBATCH --job-name=ista_learnlet_fastmri     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=ista_learnlet_fastmri_%j.out # nom du fichier de sortie
#SBATCH --error=ista_learnlet_fastmri_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/understanding-unets

. ./submission_scripts_jean_zay/env_config.sh
pip install --no-cache-dir ../fastmri-reproducible-benchmark
pip install -r ../fastmri-reproducible-benchmark/requirements.txt
export FASTMRI_DATA_DIR=$SCRATCH/


srun --ntasks=1 python ./learning_wavelets/training_scripts/ista_learnlet_fastmri_training.py -e -u -nf 32&

wait  # wait for all ccc_mprun(s) to complete.
