from GAME.Game import Game
import numpy as np


class Goofspiel(Game):
    def __init__(self, config):
        super().__init__(config)
        self.pri_feat['player1'] = 'p1_'
        self.pri_feat['player2'] = 'p2_'

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        if len(his_feat) % 3 == 1:
            return 'player1'
        else:
            return 'player2'

    def judge(self, his_feat):
        reward = 0
        i = 0
        while (i + 1) * 3 <= len(his_feat):
            if his_feat[i * 3 + 1] < his_feat[i * 3 + 2]:
                reward -= (i + 1)
            elif his_feat[i * 3 + 1] > his_feat[i * 3 + 2]:
                reward += (i + 1)
            else:
                pass
            i += 1
        return 1.0 * np.array([reward, -reward])

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        if self.get_now_player_from_his_feat(his_feat) == 'player1':
            action_list = [str(i + 1) for i in range(self.prior_state_num)]
        else:
            action_list = [str(i + 1) + '_' for i in range(self.prior_state_num)]
        i = 0
        while (i + 1) * 3 <= len(his_feat):
            if self.get_now_player_from_his_feat(his_feat) == 'player1':
                action_list.remove(his_feat[i * 3 + 1])
            else:
                action_list.remove(his_feat[i * 3 + 2] + '_')

                if len(action_list) == 2:
                    action_list1 = [str(j + 1) for j in range(self.prior_state_num)]
                    j = 0
                    while j * 3 <= len(his_feat):
                        action_list1.remove(his_feat[j * 3 + 1])
                        j += 1

                    action_list_tmp = [
                        action_list[0] + action_list1[0] + action_list[1],
                        action_list[1] + action_list1[0] + action_list[0],
                    ]
                    action_list = action_list_tmp

            i += 1

        return action_list

    def reset(self):
        pass

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:

        if his_feat[-1] == '_':
            tmp_pub_feat = his_feat
        else:
            tmp_pub_feat = his_feat[:-1]
        p1 = [str(i + 1) for i in range(self.prior_state_num)]
        p2 = [str(i + 1) for i in range(self.prior_state_num)]
        for i in range(len(tmp_pub_feat) // 3):
            p1.remove(tmp_pub_feat[3 * i + 1])
            p2.remove(tmp_pub_feat[3 * i + 2])
        return ''.join(p1) + '_' + ''.join(p2)
