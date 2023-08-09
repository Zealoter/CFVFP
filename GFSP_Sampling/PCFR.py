"""
# @Author: JuQi
# @Time  : 2022/9/29 16:43
# @E-mail: 18672750887@163.com
"""
import copy

import numpy as np
from GFSP_Sampling.GFSP import GFSPSamplingSolver


class PCFRSolver(GFSPSamplingSolver):
    def __init__(self, config: dict):
        """
        Programming optimized version of GFSPSamplingSolver
        :param config:
        """
        super().__init__(config)
        self.game.game_train_mode = 'PCFR'

    def walk_tree(self, his_feat, player_pi, pi_c):
        if self.game.game_train_mode == 'vanilla':
            return self.vanilla_walk_tree(his_feat, player_pi, pi_c)
        elif self.game.game_train_mode == 'PCFR':
            return self.PCFR_walk_tree(his_feat, player_pi, pi_c)
        return 0

    def PCFR_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if player_pi[0] == 0 and player_pi[1] == 0:
            return [0.0, 0.0]

        now_player = self.game.get_now_player_from_his_feat(his_feat)
        if now_player == 'c':
            if self.is_sampling_chance == 'all_sampling':
                r = self.PCFR_walk_tree(his_feat + self.game.get_deterministic_chance_action(his_feat), player_pi, pi_c)
            else:
                r = np.zeros(len(self.game.player_set))
                now_prob = self.game.get_chance_prob(his_feat)
                now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
                for a_i in range(len(now_action_list)):
                    tmp_r = self.PCFR_walk_tree(
                        self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                        player_pi,
                        pi_c * now_prob[a_i]
                    )
                    r += tmp_r
        else:
            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
            if len(now_action_list) == 0:
                tmp_reward = self.game.judge(his_feat)
                tmp_reward[0] = tmp_reward[0] * player_pi[1]
                tmp_reward[1] = tmp_reward[1] * player_pi[0]
                return tmp_reward * pi_c

            r = np.zeros(2)
            v = np.zeros(len(now_action_list))

            if now_player == 'player1':
                tmp_info = self.game.get_info_set('player1', his_feat)
                now_prob = player_pi[0]
                opp_prob = player_pi[1]
                now_player_id = 0
                op_player_id = 1
            else:
                tmp_info = self.game.get_info_set('player2', his_feat)
                now_prob = player_pi[1]
                opp_prob = player_pi[0]
                now_player_id = 1
                op_player_id = 0

            if tmp_info not in self.game.imm_regret.keys():
                self.game.generate_new_info_set(tmp_info, now_player, len(now_action_list))

            if opp_prob == 0:
                a_i = self.game.now_policy[tmp_info]
                tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                r[op_player_id] = tmp_r[op_player_id]
                self.game.w_his_policy[tmp_info][self.game.now_policy[tmp_info]] += self.ave_weight

            elif now_prob == 0:
                for a_i in range(len(now_action_list)):
                    tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                    v[a_i] = tmp_r[now_player_id]
                    if self.game.now_policy[tmp_info] == a_i:
                        r[now_player_id] = tmp_r[now_player_id]
                self.game.imm_regret[tmp_info] += self.ave_weight * (v - v[self.game.now_policy[tmp_info]])
                if self.is_sampling_chance == 'all_sampling':
                    self.game.now_policy[tmp_info] = np.argmax(
                        self.game.imm_regret[tmp_info] + 0.0000001 * np.random.rand(len(self.game.imm_regret[tmp_info]))
                    )

            else:
                for a_i in range(len(now_action_list)):
                    if self.game.now_policy[tmp_info] == a_i:
                        prob = 1.0
                    else:
                        prob = 0.0

                    tmp_player_pi = np.ones(2)
                    if now_player == 'player1':
                        tmp_player_pi[0] = prob
                    else:
                        tmp_player_pi[1] = prob

                    tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], tmp_player_pi, pi_c)

                    v[a_i] = tmp_r[now_player_id]
                    r[op_player_id] = r[op_player_id] + tmp_r[op_player_id]
                    r[now_player_id] = r[now_player_id] + tmp_r[now_player_id] * prob

                self.game.imm_regret[tmp_info] += self.ave_weight * (v - v[self.game.now_policy[tmp_info]])
                if self.is_sampling_chance == 'all_sampling':
                    self.P_regret_matching_strategy(tmp_info)
        return r

    def P_regret_matching_strategy(self, info):
        """
        Regret matching in sampling
        :param info:
        :return:
        """
        self.game.w_his_policy[info][self.game.now_policy[info]] += self.ave_weight
        self.game.now_policy[info] = np.argmax(
            self.game.imm_regret[info] + 0.0000001 * np.random.rand(len(self.game.imm_regret[info]))
        )

    def all_state_regret_matching_strategy(self):
        # Regret matching in full traversal mode
        for info in self.game.now_prob.keys():
            self.P_regret_matching_strategy(info)
            tmp_now_prob = self.game.now_prob[info] * self.ave_weight
            self.game.w_his_policy[info][self.game.now_policy[info]] += tmp_now_prob
            # The probability of reaching this information set
            self.game.now_prob[info] = 0
            if self.is_rm_plus:
                self.game.imm_regret[info][self.game.imm_regret[info] < 0] = 0.0
