from open_spiel.python.algorithms.psro_v2.quantalBR.nfg_to_efg import encode_gambit_file_qre

import numpy as np

meta_games = [np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]]), np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])]

encode_gambit_file_qre(meta_games)