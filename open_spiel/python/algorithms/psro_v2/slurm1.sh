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
#python combined_game.py --num_evaluation_episodes=149 --only_combine_nash=False --evaluate_nash=False --evaluate_iters=True --root_result_folder=/scratch/wellman_root/wellman1/shared_data/combined_game_source/c1 --combined_game_path=/scratch/wellman_root/wellman1/shared_data/combined_game_source/c1/combined_game2020-07-20_00-04-27/combined_game.pkl
python combined_game.py --num_evaluation_episodes=149 --only_combine_nash=True --evaluate_nash=True --evaluate_iters=False --root_result_folder=/scratch/wellman_root/wellman1/shared_data/combined_game_source/c1

