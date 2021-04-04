#!/bin/bash

#SBATCH --job-name=dqn_do
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6g
#SBATCH --time=10-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python psro_v2_example.py --game_name=kuhn_poker --n_players=2 --meta_strategy_method=CRD --oracle_type=DQN --gpsro_iterations=50 --number_training_episodes=10000 --sbatch_run=True --root_result_folder=dqn_crd_kuhn_020

