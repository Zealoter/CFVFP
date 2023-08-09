from GAME_Sampling.GameThreeCardPoker import ThreeCardPoker
from GAME_Sampling.GamePrincessAndMonster import PrincessAndMonster as PAM
from GAME_Sampling.GameLeduc import Leduc
import os
import numpy as np


def play_more_games(games_time):
    total_reward = 0
    for i_game in range(games_time):
        print('The', i_game + 1, 'game：')
        game_config = {
            'game_name'      : 'Leduc',
            'game_info'      : 'Leduc',
            'prior_state_num': 3
        }

        game = Leduc(game_config)

        game.reset()
        if np.random.rand() < 0.5:
            tmp_reward = game.game_flow(
                {
                    'player1': AI_policy,
                },
                is_show=True
            )
            total_reward -= tmp_reward
        else:
            tmp_reward = game.game_flow(
                {
                    'player2': AI_policy,
                },
                is_show=True
            )
            total_reward += tmp_reward

        print('Total utility：', total_reward)


if __name__ == '__main__':
    AI_policy = np.load(
        'AI/Leduc_PMCCFR_55910.npy',
        allow_pickle=True).item()
    play_more_games(10)
