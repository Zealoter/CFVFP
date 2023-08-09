"""
# @Author: JuQi
# @Time  : 2023/3/6 15:39
# @E-mail: 18672750887@163.com
"""
from GAME_Sampling.Game import Game
import numpy as np


class Matrix(Game):
    def __init__(self, config):
        """
        Matrix Game  Absence Sum Game and Complete Cooperative Game
        :param config:
        """
        super().__init__(config)

        self.seed = config.get('seed', 2)
        np.random.seed(self.seed)
        self.reward_matrix = np.random.normal(0, 1, size=(self.prior_state_num, self.prior_state_num))
        self.reward_matrix[0, :] += 2.5
        self.reward_matrix[:, 0] -= 2.5
        # self.reward_matrix = np.array([
        #     [0.0, 10, -1],
        #     [-1, 0, 1],
        #     [1, -1, 0]
        # ])
        # print(self.reward_matrix)

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        if len(his_feat.split('_')) == 2:
            return 'player1'
        else:
            return 'player2'

    def judge(self, his_feat: str):
        action = his_feat.split('_')
        p1_action = int(action[1]) - 1
        p2_action = int(action[2]) - 1
        tmp_reward = self.reward_matrix[p1_action, p2_action]
        return np.array([tmp_reward, -tmp_reward])

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        if len(his_feat.split('_')) <= 3:
            return [str(i + 1) + '_' for i in range(self.prior_state_num)]
        else:
            return []

        # if len(his_feat.split('_')) == 2:
        #     return ['5_', '8_', '10_', '14_', '17_', '18_', '19_', '20_']
        # elif len(his_feat.split('_')) == 3:
        #     return ['3_', '4_', '5_', '6_', '15_', '17_', '18_', '20_']
        # else:
        #     return []

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return '1_'
        else:
            return '2_'
