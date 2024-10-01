from GAME.Game import Game
import numpy as np


class Kuhn(Game):
    def __init__(self, config):
        super().__init__(config)

        self.poker = [str(i + 1) for i in range(self.prior_state_num)]
        np.random.shuffle(self.poker)

        self.pri_feat['player1'] = str(self.poker[0])
        self.pri_feat['player2'] = str(self.poker[1])

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return 'c'
        else:
            if len(self.get_pub_feat_from_his_feat(his_feat)) % 2 == 1:
                return 'player1'
            else:
                return 'player2'

    def judge(self, his_feat):
        his_feat_list = his_feat.split('_')
        if his_feat_list[-1] == 'CC':
            if his_feat_list[1] > his_feat_list[2]:
                return np.array([1.0, -1])
            else:
                return np.array([-1.0, 1])
        elif his_feat_list[-1] == 'RC' or his_feat_list[-1] == 'CRC':
            if his_feat_list[1] > his_feat_list[2]:
                return np.array([2.0, -2])
            else:
                return np.array([-2.0, 2])

        elif his_feat_list[-1] == 'RF':
            return np.array([1.0, -1])
        else:  # CRF
            return np.array([-1.0, 1])

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        if his_feat == '_':
            start_c_action_list = []
            for poker1 in range(self.prior_state_num):
                for poker2 in range(self.prior_state_num):
                    if poker1 != poker2:
                        start_c_action_list.append(self.poker[poker1] + '_' + self.poker[poker2] + '_')
            return start_c_action_list
        if his_feat[-1] == '_':
            return ['R', 'C']
        elif his_feat[-1] == 'R':
            return ['F', 'C']
        elif his_feat[-2:] == '_C':
            return ['C', 'R']
        elif his_feat[-1] == 'F':
            return []
        else:
            return []

    def get_deterministic_chance_action(self, his_feat: str) -> str:
        return self.poker[0] + '_' + self.poker[1] + '_'

    def reset(self):
        np.random.shuffle(self.poker)
        self.pri_feat['player1'] = self.poker[0]
        self.pri_feat['player2'] = self.poker[1]

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return his_feat
        else:
            pub_feat_list = his_feat.split('_')
            return '_' + pub_feat_list[3]

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        now_prob = np.ones(self.prior_state_num * (self.prior_state_num - 1))
        now_prob = now_prob / np.sum(now_prob)
        return now_prob

    def get_info_set(self, player_id, his_feat):
        his_feat_list = his_feat.split('_')
        if player_id == 'player1':
            return his_feat_list[1] + self.get_pub_feat_from_his_feat(his_feat)
        else:
            return his_feat_list[2] + self.get_pub_feat_from_his_feat(his_feat)
