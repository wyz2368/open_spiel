#!/bin/bash

#SBATCH --job-name=quiesce_d
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40g
#SBATCH --time=13-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python psro_v2_quiesce.py --quiesce_regret_threshold=0.0 --RD_regret_threshold=0.5 --RD_regularization=False --quiesce_control=True --game_name=leduc_poker --n_players=3 --meta_strategy_method=prd --oracle_type=BR --gpsro_iterations=50 --number_training_episodes=10000 --sbatch_run=True --root_result_folder=br_do_quiesce_supcontrol

