from open_spiel.python.algorithms.psro_v2.quantalBR.nfg_to_efg import encode_gambit_file_qre

import numpy as np

# meta_games = [np.array([[1,2,3],
#                         [4,5,6],
#                         [7,8,9]]), np.array([[1,2,3],
#                         [4,5,6],
#                         [7,8,9]])]

meta_games = [np.array([[1, -1, 0],
                        [-1, 1, 0],
                        [0, 0, 0]]),
              np.array([[-1, 1, 0],
                        [1, -1, 0],
                        [0, 0, 0]])]

# meta_games = [np.array([[0.1, 0, 0],
#                         [0.2, 3, 0],
#                         [0, 0, 6]]),
#               np.array([[0.1, 0, 0],
#                         [0.2, 2, 0],
#                         [0, 0, 6]])]

encode_gambit_file_qre(meta_games)