"""
# @Author: JuQi
# @Time  : 2023/3/6 15:39
# @E-mail: 18672750887@163.com
"""
from GAME_Sampling.Game import Game
import numpy as np


class PrincessAndMonster(Game):
    def __init__(self, config):
        super().__init__(config)
        self.boundary_length = config.get('boundary_length', 3)
        self.boundary_width = config.get('boundary_width', 3)
        self.game_time = config.get('prior_state_num', 2)
        self.boundary_point = config.get('boundary_point', ['0,0_', '0,2_'])

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return 'player1'
        else:
            ob_his_feat_list = his_feat.split('_')
            if len(ob_his_feat_list) % 2 == 1:
                return 'player2'
            else:
                return 'player1'

    def judge(self, his_feat):
        his_feat_list = his_feat.split('_')
        r = (len(his_feat_list) - 2) / 2.0
        return np.array([-r, r])

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        ob_his_feat_list = his_feat.split('_')
        action_list = []
        if his_feat == '_' or len(ob_his_feat_list) == 3:
            for x in range(self.boundary_length):
                for y in range(self.boundary_width):
                    action_list.append(str(x) + ',' + str(y) + '_')
        else:
            if len(ob_his_feat_list) == self.game_time * 2 + 2:
                pass
            else:
                if len(ob_his_feat_list) % 2 == 0 and ob_his_feat_list[-2] == ob_his_feat_list[-3]:
                    return []
                else:
                    action_list = [ob_his_feat_list[-3] + '_']
                    tmp_x = int(ob_his_feat_list[-3][0])
                    tmp_y = int(ob_his_feat_list[-3][2])
                    if tmp_x != 0:
                        action_list.append(str(tmp_x - 1) + ',' + str(tmp_y) + '_')
                    if tmp_x != self.boundary_length - 1:
                        action_list.append(str(tmp_x + 1) + ',' + str(tmp_y) + '_')
                    if tmp_y != 0:
                        action_list.append(str(tmp_x) + ',' + str(tmp_y - 1) + '_')
                    if tmp_y != self.boundary_width - 1:
                        action_list.append(str(tmp_x) + ',' + str(tmp_y + 1) + '_')
        for i_point in self.boundary_point:
            if i_point in action_list:
                action_list.remove(i_point)

        return action_list

    def get_info_set(self, player_id, his_feat):
        his_feat_list = his_feat.split('_')

        if len(his_feat_list) == 2:
            return 'M_'
        elif len(his_feat_list) == 3:
            return 'P_'
        else:
            his_feat_list = his_feat_list[1:-1]
            if player_id == 'player1':
                tmp_info = 'M_'
                for i in range(len(his_feat_list)):
                    if i % 2 == 0:
                        tmp_info += his_feat_list[i]
                        tmp_info += '_'
                return tmp_info
            else:
                tmp_info = 'P_'
                for i in range(len(his_feat_list)):
                    if i % 2 == 1:
                        tmp_info += his_feat_list[i]
                        tmp_info += '_'
                return tmp_info
