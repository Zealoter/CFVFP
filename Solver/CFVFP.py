import numpy as np
from Solver.CFR import CFRSolver


class CFVFPSolver(CFRSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self.game.game_train_mode = 'CFVFP'

    def walk_tree(self, his_feat, player_pi, pi_c):
        return self.CFVFP_walk_tree(his_feat, player_pi, pi_c)

    def CFVFP_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if np.sum(player_pi) == 0:
            return [0.0, 0.0]

        now_player = self.game.get_now_player_from_his_feat(his_feat)
        if now_player == 'c':
            if self.sampling_mode == 'sampling':
                r = self.CFVFP_walk_tree(his_feat + self.game.get_deterministic_chance_action(his_feat), player_pi, pi_c)
            else:
                r = np.zeros(len(self.game.player_set))
                now_prob = self.game.get_chance_prob(his_feat)
                now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
                for a_i in range(len(now_action_list)):
                    tmp_r = self.CFVFP_walk_tree(
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

            if tmp_info not in self.game.his_regret.keys():
                self.game.generate_new_info_set(tmp_info, now_player, len(now_action_list))

            if opp_prob == 0:
                a_i = self.game.now_policy[tmp_info]
                tmp_r = self.CFVFP_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                r[op_player_id] = tmp_r[op_player_id]
                self.game.w_his_policy[tmp_info][self.game.now_policy[tmp_info]] += self.ave_weight

            elif now_prob == 0:
                for a_i in range(len(now_action_list)):
                    tmp_r = self.CFVFP_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                    v[a_i] = tmp_r[now_player_id]
                    if self.game.now_policy[tmp_info] == a_i:
                        r[now_player_id] = tmp_r[now_player_id]
                self.game.his_regret[tmp_info] += self.ave_weight * v
                if self.sampling_mode == 'sampling':
                    self.game.now_policy[tmp_info] = np.argmax(
                        self.game.his_regret[tmp_info]
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

                    tmp_r = self.CFVFP_walk_tree(his_feat + now_action_list[a_i], tmp_player_pi, pi_c)

                    v[a_i] = tmp_r[now_player_id]
                    r[op_player_id] = r[op_player_id] + tmp_r[op_player_id]
                    r[now_player_id] = r[now_player_id] + tmp_r[now_player_id] * prob

                self.game.his_regret[tmp_info] += self.ave_weight * v
                self.game.w_his_policy[tmp_info][self.game.now_policy[tmp_info]] += self.ave_weight
                if self.sampling_mode == 'sampling':
                    self.game.now_policy[tmp_info] = np.argmax(
                        self.game.his_regret[tmp_info]
                    )
        return r

    def all_state_regret_matching_strategy(self):
        # 在全遍历模式下的遗憾匹配
        for info in self.game.now_prob.keys():
            self.game.w_his_policy[info][self.game.now_policy[info]] += self.ave_weight
            self.game.now_policy[info] = np.argmax(
                self.game.his_regret[info]
            )
            tmp_now_prob = self.game.now_prob[info] * self.ave_weight
            self.game.w_his_policy[info][self.game.now_policy[info]] += tmp_now_prob
            # 到达这个信息集的概率
            self.game.now_prob[info] = 0
            if self.is_rm_plus:
                self.game.his_regret[info][self.game.his_regret[info] < 0] = 0.0
