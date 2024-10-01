from GAME.GameKuhn import Kuhn
import numpy as np


class KuhnNPot(Kuhn):
    def __init__(self, config):
        super().__init__(config)
        self.y_pot = config.get('y_pot', 5)
        self.z_len = config.get('z_len', 3)

        self.pot_symbol = [chr(i) for i in range(33, 128)]
        self.pot_symbol.remove('C')
        self.pot_symbol.remove('F')
        self.pot_symbol.remove('_')
        self.pot_symbol.remove('\'')
        self.pot_symbol.remove('\"')
        self.pot_symbol = self.pot_symbol[:self.y_pot]
        self.poker = [str(i + 1) for i in range(self.prior_state_num)]
        np.random.shuffle(self.poker)

        self.pri_feat['player1'] = str(self.poker[0])
        self.pri_feat['player2'] = str(self.poker[1])

    def judge(self, his_feat):
        his_feat_act = his_feat.split('_')[-1]
        if his_feat[-1] == 'F':
            if len(his_feat_act) <= 2:
                money = 1
            else:
                if his_feat[-3] == 'C':
                    money = 1
                else:
                    # money = (self.pot_symbol.index(his_feat[-3]) + 1.0) / self.y_pot * 100
                    money = 2 ** self.pot_symbol.index(his_feat[-3])

            if self.get_now_player_from_his_feat(his_feat) == 'player1':
                return np.array([1.0 * money, -money])
            else:
                return np.array([-1.0 * money, money])
        elif his_feat[-1] == 'C' and his_feat[-2] != '_':
            if his_feat[-2] == 'C':
                money = 1
            else:
                # money = (self.pot_symbol.index(his_feat[-2]) + 1.0) / self.y_pot * 100
                money = 2 ** self.pot_symbol.index(his_feat[-2])

            his_feat_list = his_feat.split('_')
            if his_feat_list[1] > his_feat_list[2]:
                return np.array([1.0 * money, -money])
            else:
                return np.array([-1.0 * money, money])
        else:
            return np.array([0.0, 0])

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        his_feat_act = his_feat.split('_')[-1]
        if his_feat == '_':
            start_c_action_list = []
            for poker1 in range(self.prior_state_num):
                for poker2 in range(self.prior_state_num):
                    if poker1 != poker2:
                        start_c_action_list.append(self.poker[poker1] + '_' + self.poker[poker2] + '_')
            return start_c_action_list
        if his_feat[-1] == 'C' and his_feat[-2] != '_':
            return []
        elif his_feat[-1] == 'F':
            return []
        elif (len(his_feat_act) >= self.z_len and his_feat_act[0] != 'C') or len(
                his_feat_act) >= self.z_len + 1:  # 最多加注3次
            return ['F', 'C']
        elif his_feat[-1] == '_' or his_feat[-2:] == '_C':
            return self.pot_symbol + ['C']
        else:
            tmp_pot = his_feat[-1]
            tmp_pot_index = self.pot_symbol.index(tmp_pot)
            if tmp_pot_index == self.y_pot:
                return ['F', 'C']
            else:
                return self.pot_symbol[tmp_pot_index + 1:] + ['F', 'C']
