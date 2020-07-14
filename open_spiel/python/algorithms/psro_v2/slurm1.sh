#!/bin/bash

#SBATCH --job-name=combined_game
#SBATCH --mail-user=qmaai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20g
#SBATCH --time=02-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
#python se_example.py --game_name=leduc_poker --n_players=2 --switch_blocks=True --standard_regret=True --fast_oracle_period=1 --slow_oracle_period=3 --meta_strategy_method=uniform --oracle_type=BR --gpsro_iterations=110 --number_training_episodes=10000 --sbatch_run=True --root_result_folder=br_block
python combined_game.py -num_evaluation_episodes=149 --only_combine_nash=True -root_result_folder=/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/leduc_combined_game_do

