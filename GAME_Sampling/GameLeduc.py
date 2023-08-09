"""
# @Author: JuQi
# @Time  : 2023/3/6 15:39
# @E-mail: 18672750887@163.com
"""
from GAME_Sampling.Game import Game
import numpy as np


class Leduc(Game):
    def __init__(self, config):
        super().__init__(config)
        self.poker = []
        for i in range(self.prior_state_num):
            self.poker.append(str(i + 1))
            self.poker.append(str(i + 1))
        np.random.shuffle(self.poker)

        self.pri_feat['player1'] = str(self.poker[0])
        self.pri_feat['player2'] = str(self.poker[1])

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return 'c'
        else:
            tmp_h = his_feat.split('_')
            if len(tmp_h) == 4:
                if tmp_h[-1][-2:] == 'CC':
                    return 'c'
                elif len(tmp_h[-1]) >= 2 and tmp_h[-1][-1] == 'C' and tmp_h[-1][-2] != '_':
                    return 'c'
                elif len(tmp_h[-1]) % 2 == 1:
                    return 'player2'
                else:
                    return 'player1'
            else:
                if (len(tmp_h[3]) + len(tmp_h[5])) % 2 == 1:
                    return 'player2'
                else:
                    return 'player1'

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        if his_feat == '_':
            start_c_action_list = []
            for poker1 in range(1, self.prior_state_num + 1):
                for poker2 in range(1, self.prior_state_num + 1):
                    start_c_action_list.append(str(poker1) + '_' + str(poker2) + '_')
            return start_c_action_list
        if his_feat.count('_') == 3 and his_feat[-1] == '_':
            return ['F', 'C', 'R']
        elif his_feat[-1] == '_':
            return ['R', 'C']
        elif his_feat[-2:] == 'RR':
            return ['F', 'C']
        elif his_feat[-1] == 'R':
            return ['F', 'C', 'R']
        elif his_feat[-2:] == '_C':
            return ['C', 'R']
        elif his_feat[-1] == 'F':
            return []
        elif his_feat[-1] == 'C':
            tmp_h = his_feat.split('_')
            if len(tmp_h) == 4:
                if tmp_h[1] != tmp_h[2]:
                    return ['_' + str(i) + '_' for i in range(1, self.prior_state_num + 1)]
                else:
                    tmp_act_list = ['_' + str(i) + '_' for i in range(1, self.prior_state_num + 1)]
                    tmp_act_list.remove('_' + tmp_h[1] + '_')
                    return tmp_act_list
            else:
                return []
        else:
            return []

    def judge(self, his_feat):
        if his_feat[-2:] == '_F':
            return np.array([-1.0, 1.0])
        now_player = self.get_now_player_from_his_feat(his_feat)
        tmp_h = his_feat.split('_')
        money = 2.0
        money = money + 2 * tmp_h[3].count('R')
        if len(tmp_h) == 6:
            money = money + 4 * tmp_h[5].count('R')

        if tmp_h[-1] == 'F':
            if now_player == 'player1':
                return np.array([money, -money])
            elif now_player == 'player2':
                return np.array([-money, money])

        elif tmp_h[-1][-2:] == 'RF':
            if len(tmp_h) == 6:
                money -= 4
            else:
                money -= 2
            if now_player == 'player1':
                return np.array([money, -money])
            elif now_player == 'player2':
                return np.array([-money, money])

        elif tmp_h[-1][-2:] == 'CC' or tmp_h[-1][-2:] == 'RC':
            if tmp_h[1] == tmp_h[4]:
                return np.array([money, -money])
            elif tmp_h[2] == tmp_h[4]:
                return np.array([-money, money])
            elif tmp_h[1] < tmp_h[2]:
                return np.array([-money, money])
            elif tmp_h[1] > tmp_h[2]:
                return np.array([money, -money])
            else:
                return np.array([0.0, 0.0])

    def reset(self):
        np.random.shuffle(self.poker)
        self.pri_feat['player1'] = str(self.poker[0])
        self.pri_feat['player2'] = str(self.poker[1])

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        if his_feat == '_':
            start_c_action_prob = []
            for poker1 in range(self.prior_state_num):
                for poker2 in range(self.prior_state_num):
                    if poker1 != poker2:
                        start_c_action_prob.append(2)
                    else:
                        start_c_action_prob.append(1)
            now_prob = np.array(start_c_action_prob)
        else:
            tmp_h = his_feat.split('_')
            if tmp_h[1] != tmp_h[2]:
                now_prob = np.ones(self.prior_state_num)
                now_prob *= 2
                now_prob[int(tmp_h[1]) - 1] -= 1
                now_prob[int(tmp_h[2]) - 1] -= 1
            else:
                now_prob = np.ones(self.prior_state_num - 1)
        now_prob = now_prob / np.sum(now_prob)
        return now_prob

    def get_deterministic_chance_action(self, his_feat: str) -> str:
        if his_feat == '_':
            return self.poker[0] + '_' + self.poker[1] + '_'
        else:
            return '_' + self.poker[2] + '_'

    def get_info_set(self, player_id, his_feat):
        his_feat_list = his_feat.split('_')
        if player_id == 'player1':
            return his_feat_list[1] + self.get_pub_feat_from_his_feat(his_feat)
        else:
            return his_feat_list[2] + self.get_pub_feat_from_his_feat(his_feat)

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return his_feat
        else:
            pub_feat_list = his_feat.split('_')
            ob_his_fear = '_'.join(pub_feat_list[3:])
            return '_' + ob_his_fear
