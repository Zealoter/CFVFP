import numpy as np
import os
import time
import csv
import copy
from GAME_Sampling.GameKuhn import Kuhn
from GAME_Sampling.GameLeduc import Leduc
from GAME_Sampling.GameLeduc3Pot import Leduc3Pot
from GAME_Sampling.GameLeduc5Pot import Leduc5Pot
from GAME_Sampling.GameKuhnNPot import KuhnNPot

if __name__ == '__main__':
    file_list = os.listdir('league')
    print(file_list)
    league = []
    for i_policy_name in file_list:
        league.append(np.load('league' + '/' + i_policy_name, allow_pickle=True).item())

    league_len = len(league)

    game_config = {
        'game_name'      : 'Leduc',
        'game_info'      : '12',
        'prior_state_num': 12,
        'y_pot'          : 5,
        'z_len'          : 3,
    }
    if game_config['game_name'] == 'Kuhn':
        game = Kuhn(game_config)
    elif game_config['game_name'] == 'Leduc':
        game = Leduc(game_config)
    elif game_config['game_name'] == 'Leduc3Pot':
        game = Leduc3Pot(game_config)
    elif game_config['game_name'] == 'Leduc5Pot':
        game = Leduc5Pot(game_config)
    elif game_config['game_name'] == 'KuhnNPot':
        game = KuhnNPot(game_config)
    else:
        pass

    reward_matrix1 = np.zeros((league_len, league_len))
    reward_matrix2 = np.zeros((league_len, league_len))
    for p_i in range(league_len):
        print(p_i)
        for p_j in range(league_len):
            for i_game in range(100000):
                game.reset()
                tmp_reward = game.game_flow({
                    'player1': league[p_i],
                    'player2': league[p_j],
                })
                reward_matrix1[p_i, p_j] += tmp_reward[0]
    print(reward_matrix1 / 100000)
