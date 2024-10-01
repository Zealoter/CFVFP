from GAME.GameLeduc import Leduc
import numpy as np


class Leduc3Pot(Leduc):
    def __init__(self, config):
        super().__init__(config)

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        if his_feat == '_':
            start_c_action_list = []
            for poker1 in range(1, self.prior_state_num + 1):
                for poker2 in range(1, self.prior_state_num + 1):
                    start_c_action_list.append(str(poker1) + '_' + str(poker2) + '_')
            return start_c_action_list

        pub_feat = self.get_pub_feat_from_his_feat(his_feat)
        if pub_feat[-1] == 'F':
            return []
        elif pub_feat == '_':
            return ['F', 'C', 'R', 'S', 'T']
        elif pub_feat[-1] == '_':
            return ['C', 'R', 'S', 'T']
        elif pub_feat[-1] != 'C' and pub_feat[-2] != 'C':
            return ['F', 'C']
        elif pub_feat[-1] == 'R':
            return ['F', 'C', 'R', 'S', 'T']
        elif pub_feat[-1] == 'S':
            return ['F', 'C', 'S', 'T']
        elif pub_feat[-1] == 'T':
            return ['F', 'C', 'T']
        elif pub_feat[-2:] == '_C':
            return ['C', 'R', 'S', 'T']

        elif pub_feat[-1] == 'C':
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
        now_player = self.get_now_player_from_his_feat(his_feat)
        pub_feat = self.get_pub_feat_from_his_feat(his_feat)
        if pub_feat == '_F':
            return np.array([-1.0, 1.0])
        tmp_h = his_feat.split('_')
        money = 2.0
        money = money + 2 * tmp_h[3].count('R')
        money = money + 4 * tmp_h[3].count('S')
        money = money + 6 * tmp_h[3].count('T')

        if len(tmp_h) == 6:
            money = money + 4 * tmp_h[5].count('R')
            money = money + 8 * tmp_h[5].count('S')
            money = money + 16 * tmp_h[5].count('T')

        if tmp_h[-1] == 'F':
            if now_player == 'player1':
                return np.array([money, -money])
            elif now_player == 'player2':
                return np.array([-money, money])

        elif tmp_h[-1][-1] == 'F':
            if len(tmp_h) == 6:
                if tmp_h[-1][-2] == 'R':
                    money -= 4
                elif tmp_h[-1][-2] == 'S':
                    money -= 8
                else:
                    money -= 16

            else:
                if tmp_h[-1][-2] == 'R':
                    money -= 2
                elif tmp_h[-1][-2] == 'S':
                    money -= 4
                else:
                    money -= 8

            if now_player == 'player1':
                return np.array([money, -money])
            elif now_player == 'player2':
                return np.array([-money, money])

        elif len(tmp_h[-1]) >= 2 and tmp_h[-1][-1] == 'C':
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
