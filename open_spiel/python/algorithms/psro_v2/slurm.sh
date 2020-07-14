#!/bin/bash

#SBATCH --job-name=combined_game
#SBATCH --mail-user=qmaai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=80g
#SBATCH --time=05-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=largemem

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python combined_game.py --evaluate_nash=False --root_result_folder=/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/folder1 --sims_per_entry=10 --num_evaluation_episodes=10
