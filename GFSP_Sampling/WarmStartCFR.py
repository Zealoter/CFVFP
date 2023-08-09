
import numpy as np
from GFSP_Sampling.PCFR import PCFRSolver
import os
import copy


class WSCFRSolver(PCFRSolver):
    def __init__(self, config: dict):
        """
        Programming optimized version of GFSPSamplingSolver
        :param config:
        """
        super().__init__(config)
        self.teacher_policy = None
        self.result_file_path += 'WS'

    def walk_tree(self, his_feat, player_pi, pi_c):
        if self.game.game_train_mode == 'PCFR':
            r = self.PCFR_walk_tree(his_feat, player_pi, pi_c)
            if self.node_touched >= 1000:
                self.node_touched = 0
                self.itr_num = 0
                self.game.game_train_mode = 'vanilla'
                self.is_sampling_chance = 'no_sampling'
                self.rm_mode = 'vanilla'
                self.is_rm_plus = True
                self.teacher_policy = self.game.get_his_mean_policy()
                self.total_weight = 0
                self.p1_total_reward = 0
                self.last_itr_node_touched = 0
                self.first_log = True
                for i_key in self.teacher_policy.keys():
                    self.teacher_policy[i_key] = self.teacher_policy[i_key] > 0.001
                self.itr_num = 0
                self.log_threshold = self.log_interval
                self.game.w_his_policy = {}
                self.game.now_policy = {}
                self.game.imm_regret = {}
                self.game.now_prob = {}
                for p in self.game.player_set:
                    self.game.info_set_list[p] = []
                self.result_file_path = self.result_file_path[:-2]
                os.makedirs(self.result_file_path)
            return r

        elif self.game.game_train_mode == 'vanilla':
            return self.WS_walk_tree(his_feat, player_pi, pi_c)
        return 0

    def WS_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if pi_c == 0 or np.sum(player_pi) < 1.0e-20:
            return np.zeros(len(self.game.player_set))

        r = np.zeros(len(self.game.player_set))

        if self.game.get_now_player_from_his_feat(his_feat) == 'c':
            now_prob = self.game.get_chance_prob(his_feat)
            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)

            for a_i in range(len(now_action_list)):
                tmp_r = self.WS_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    player_pi,
                    pi_c * now_prob[a_i]
                )
                r += tmp_r
        else:
            now_player = self.game.get_now_player_from_his_feat(his_feat)
            tmp_info = self.game.get_info_set(now_player, his_feat)
            now_player_id = self.game.player_set.index(now_player)
            now_prob = player_pi[now_player_id]

            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)

            if tmp_info in self.teacher_policy.keys():
                new_now_action_list = []
                for tmp_i in range(len(now_action_list)):
                    if self.teacher_policy[tmp_info][tmp_i]:
                        new_now_action_list.append(now_action_list[tmp_i])
                now_action_list = new_now_action_list

            v = np.zeros(len(now_action_list))

            if len(now_action_list) == 0:
                tmp_reward = self.game.judge(his_feat)
                for i in range(len(self.game.player_set)):
                    for j in range(len(self.game.player_set)):
                        if i != j:
                            tmp_reward[i] = tmp_reward[i] * player_pi[j]
                return tmp_reward * pi_c

            if tmp_info not in self.game.imm_regret.keys():
                self.game.generate_new_info_set(tmp_info, now_player, len(now_action_list))

            if self.is_sampling_chance == 'no_sampling':
                # The probability of reaching the current node should be read during full traversal
                self.game.now_prob[tmp_info] = now_prob
            else:
                self.regret_matching_strategy(tmp_info)
                tmp_now_prob = now_prob * self.game.now_policy[tmp_info]
                tmp_now_prob = tmp_now_prob * self.ave_weight
                self.game.w_his_policy[tmp_info] += tmp_now_prob

            for a_i in range(len(now_action_list)):
                new_player_pi = copy.deepcopy(player_pi)
                new_player_pi[now_player_id] = new_player_pi[now_player_id] * self.game.now_policy[tmp_info][a_i]
                tmp_r = self.WS_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    new_player_pi,
                    pi_c
                )

                v[a_i] += tmp_r[now_player_id]
                for i in range(len(self.game.player_set)):
                    if i != now_player_id:
                        r[i] = r[i] + tmp_r[i]
                r[now_player_id] = r[now_player_id] + tmp_r[now_player_id] * self.game.now_policy[tmp_info][a_i]

            self.update_regrets(tmp_info, v)

        return r

    def WSPolicy2OriginalPolicy(self, ws_policy: dict) -> dict:
        original_policy = {}
        for info_set in ws_policy:
            if info_set in self.teacher_policy.keys():
                self.game.imm_regret[info_set] = np.zeros(len(self.teacher_policy[info_set]))
                self.game.now_policy[info_set] = np.zeros(len(self.teacher_policy[info_set]))

                original_policy[info_set] = np.zeros(len(self.teacher_policy[info_set]))
                i_act_org = 0
                for i_act in range(len(self.teacher_policy[info_set])):
                    if self.teacher_policy[info_set][i_act]:
                        original_policy[info_set][i_act] = ws_policy[info_set][i_act_org]
                        i_act_org += 1
            else:
                original_policy[info_set] = ws_policy[info_set]

        return original_policy

    def get_epsilon(self, appr_ne_policy: dict):
        # p1 BR value
        self.is_sampling_chance = 'no_sampling'
        appr_ne_policy = self.WSPolicy2OriginalPolicy(appr_ne_policy)
        self.game.now_policy = copy.deepcopy(appr_ne_policy)

        game_value = self.epsilon_walk_tree('_', np.ones(len(self.game.player_set)), 1.0)
        fix_imm_regret = copy.deepcopy(self.game.imm_regret)
        for info in self.game.info_set_list['player1']:
            br_act = np.argmax(fix_imm_regret[info])
            self.game.now_policy[info] = np.zeros_like(fix_imm_regret[info])
            self.game.now_policy[info][br_act] = 1.0
        p1_br_value = self.epsilon_walk_tree('_', np.ones(len(self.game.player_set)), 1.0)

        self.game.now_policy = copy.deepcopy(appr_ne_policy)

        for info in self.game.info_set_list['player2']:
            br_act = np.argmax(fix_imm_regret[info])
            self.game.now_policy[info] = np.zeros_like(fix_imm_regret[info])
            self.game.now_policy[info][br_act] = 1.0
        p2_br_value = self.epsilon_walk_tree('_', np.ones(len(self.game.player_set)), 1.0)

        epsilon = p1_br_value[0] + p2_br_value[1]
        return epsilon, game_value

    def all_state_regret_matching_strategy(self):
        for info in self.game.now_prob.keys():
            if self.game.game_train_mode == 'vanilla':
                self.regret_matching_strategy(info)
                tmp_now_prob = self.game.now_prob[info] * self.game.now_policy[info]
                tmp_now_prob = tmp_now_prob * self.ave_weight
                self.game.w_his_policy[info] += tmp_now_prob

            elif self.game.game_train_mode == 'PCFR':
                self.P_regret_matching_strategy(info)
                tmp_now_prob = self.game.now_prob[info] * self.ave_weight
                self.game.w_his_policy[info][self.game.now_policy[info]] += tmp_now_prob

            self.game.now_prob[info] = 0
            if self.is_rm_plus:
                self.game.imm_regret[info][self.game.imm_regret[info] < 0] = 0.0
