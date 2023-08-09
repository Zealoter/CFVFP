
import numpy as np
import os
import time
import csv
import copy


class GFSPSamplingSolver(object):
    def __init__(self, config: dict):
        """
        Sampling framework
        :param config:
        """
        self.game = config.get('game')
        self.total_exp_name = config.get('total_exp_name')
        self.game_name = self.game.game_name
        self.game_info = config.get('game_info', '')
        self.is_show_policy = config.get('is_show_policy', False)
        now_path_str = os.getcwd()
        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.result_file_path = ''.join(
            [
                now_path_str,
                '/logGFSPSampling/',
                self.total_exp_name,
                '/',
                self.game_info,
                '/',
                str(config.get('No.')),
                '_',
                now_time_str
            ]
        )
        self.rm_mode = config.get('rm_mode', 'vanilla')
        self.rm_eta = config.get('rm_eta', 1)
        self.cur_ave_weight = 1
        if self.rm_mode == 'eta_fix':
            self.cur_eta = self.rm_eta

        self.is_rm_plus = config.get('is_rm_plus', False)
        self.is_sampling_chance = config.get('is_sampling_chance', 'all_sampling')
        self.node_touched = 0
        self.train_mode = config.get('train_mode', 'fix_itr')

        self.total_train_constraint = config.get('total_train_constraint', 1000)
        self.log_interval = config.get('log_interval', 100)
        self.log_interval_mode = config.get('log_interval_mode')
        self.ave_mode = config.get('ave_mode', 'vanilla')
        self.log_mode = config.get('log_mode', 'normal')
        self.log_state_start_num = config.get('log_state_start_num', 1000)
        self.start_time = 0
        self.itr_num = 0
        self.total_weight = 0
        self.p1_total_reward = 0
        self.pure_policy_num = 0
        self.last_itr_node_touched = 0
        self.moving_ave_pure_policy_rate = 0
        self.first_log = True
        self.ave_weight = 0
        self.log_threshold = self.log_interval

    def save_model(self, episode):
        """
        Save policy
        :param episode: Training times
        :return:
        """
        tmp_policy = self.game.get_his_mean_policy()
        np.save(self.result_file_path + '/' + str(episode) + '.npy', tmp_policy)

    def log_model(self, itr: int, train_time: float = 0.0):
        """
        Record data during training
        :param train_time: Time spent on training;
        :param itr: Training times
        :return:
        """
        sum_imm_regret_per_player = self.game.get_sum_imm_regret()  # 全部信息集上的即时遗憾值
        weighted_mean_imm_regret = sum_imm_regret_per_player / self.total_weight

        epsilon = 0
        log_info = {
            'itr'                   : itr,
            'train_time(ms)'        : train_time * 1000,
            'epsilon'               : epsilon,
            'mean_imm_regret'       : np.sum(weighted_mean_imm_regret),
            'node_touched'          : self.node_touched,
            'sum_imm_regret'        : np.sum(sum_imm_regret_per_player),
            'mean_reward'           : self.p1_total_reward / itr,
            'perfect_BR1'           : weighted_mean_imm_regret[0],
            'perfect_BR2'           : weighted_mean_imm_regret[1],
            'last_time_node_touched': self.node_touched - self.last_itr_node_touched,
            'pure_policy_ratio'     : self.pure_policy_num / len(self.game.imm_regret)
        }

        self.save_model(self.itr_num)
        with open(self.result_file_path + '/tmp.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_info.keys())
            if self.first_log:
                writer.writeheader()
            writer.writerow(log_info)
        return

    def get_epsilon(self, appr_ne_policy: dict):
        # p1 BR value
        self.is_sampling_chance = 'no_sampling'
        self.game.now_policy = copy.deepcopy(appr_ne_policy)
        for i_key in self.game.imm_regret.keys():
            self.game.imm_regret[i_key] *= 0

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

    def walk_tree(self, his_feat, player_pi, pi_c):
        return self.vanilla_walk_tree(his_feat, player_pi, pi_c)

    def vanilla_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if pi_c == 0 or np.sum(player_pi) < 1.0e-20:
            return np.zeros(len(self.game.player_set))

        r = np.zeros(len(self.game.player_set))

        if self.game.get_now_player_from_his_feat(his_feat) == 'c':
            if self.is_sampling_chance == 'all_sampling':
                r = self.vanilla_walk_tree(
                    self.game.get_next_his_feat(his_feat, self.game.get_deterministic_chance_action(his_feat)),
                    player_pi,
                    pi_c
                )
            else:
                now_prob = self.game.get_chance_prob(his_feat)
                now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
                if len(now_action_list) == 0:
                    tmp_reward = self.game.judge(his_feat)
                    for i in range(len(self.game.player_set)):
                        for j in range(len(self.game.player_set)):
                            if i != j:
                                tmp_reward[i] = tmp_reward[i] * player_pi[j]
                    return tmp_reward * pi_c

                for a_i in range(len(now_action_list)):
                    tmp_r = self.vanilla_walk_tree(
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
            if len(now_action_list) == 0:
                tmp_reward = self.game.judge(his_feat)
                for i in range(len(self.game.player_set)):
                    for j in range(len(self.game.player_set)):
                        if i != j:
                            tmp_reward[i] = tmp_reward[i] * player_pi[j]
                return tmp_reward * pi_c
            v = np.zeros(len(now_action_list))

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
                tmp_r = self.vanilla_walk_tree(
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

    def epsilon_walk_tree(self, his_feat, player_pi, pi_c):
        if pi_c == 0 or np.sum(player_pi) < 1.0e-20:
            return np.zeros(len(self.game.player_set))
        now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
        if len(now_action_list) == 0:
            tmp_reward = self.game.judge(his_feat)
            for i in range(len(self.game.player_set)):
                for j in range(len(self.game.player_set)):
                    if i != j:
                        tmp_reward[i] = tmp_reward[i] * player_pi[j]
            return tmp_reward * pi_c

        r = np.zeros(len(self.game.player_set))

        if self.game.get_now_player_from_his_feat(his_feat) == 'c':
            now_prob = self.game.get_chance_prob(his_feat)
            for a_i in range(len(now_action_list)):
                tmp_r = self.epsilon_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    player_pi,
                    pi_c * now_prob[a_i]
                )
                r += tmp_r
        else:
            now_player = self.game.get_now_player_from_his_feat(his_feat)
            tmp_info = self.game.get_info_set(now_player, his_feat)
            now_player_id = self.game.player_set.index(now_player)

            v = np.zeros(len(now_action_list))

            if tmp_info not in self.game.now_policy.keys():
                self.game.now_policy[tmp_info] = np.ones(len(now_action_list)) / len(now_action_list)
            if tmp_info not in self.game.imm_regret.keys():
                self.game.imm_regret[tmp_info] = np.zeros(len(now_action_list))

            for a_i in range(len(now_action_list)):
                new_player_pi = copy.deepcopy(player_pi)
                new_player_pi[now_player_id] = new_player_pi[now_player_id] * self.game.now_policy[tmp_info][a_i]
                tmp_r = self.epsilon_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    new_player_pi,
                    pi_c
                )

                v[a_i] += tmp_r[now_player_id]
                for i in range(len(self.game.player_set)):
                    if i != now_player_id:
                        r[i] = r[i] + tmp_r[i]
                r[now_player_id] = r[now_player_id] + tmp_r[now_player_id] * self.game.now_policy[tmp_info][a_i]

            tmp_regret = v - np.sum(self.game.now_policy[tmp_info] * v)
            self.game.imm_regret[tmp_info] += tmp_regret
        return r

    def update_regrets(self, info, v):
        tmp_regret = v - np.sum(self.game.now_policy[info] * v)
        if self.is_rm_plus:
            tmp_regret = tmp_regret * self.cur_ave_weight
        else:
            tmp_regret = tmp_regret * self.ave_weight
        self.game.imm_regret[info] += tmp_regret
        if self.is_rm_plus:
            if self.is_sampling_chance != 'no_sampling':
                self.game.imm_regret[info][self.game.imm_regret[info] < 0] = 0.0

    def get_ave_weight(self):
        if self.ave_mode == 'vanilla':
            return self.cur_ave_weight
        elif self.ave_mode == 'log':
            return np.log10(self.cur_ave_weight * self.itr_num + 1)
        elif self.ave_mode == 'liner':
            return self.cur_ave_weight * self.itr_num
        elif self.ave_mode == 'square':
            return self.cur_ave_weight * self.itr_num * self.itr_num
        else:
            print('error')
            return 0

    def regret_matching_strategy(self, info):
        """
        Regret matching in sampling
        :param info:
        :return:
        """
        tmp_r = copy.deepcopy(self.game.imm_regret[info])
        tmp_r[tmp_r < 0] = 0.0
        if np.max(tmp_r) > 0 and np.max(tmp_r) + 0.0000001 > np.sum(tmp_r):
            self.pure_policy_num += 1
        if np.sum(tmp_r) < 1.0e-20:
            if self.is_sampling_chance == 'all_sampling':
                tmp_act = np.random.randint(len(self.game.now_policy[info]))
                self.game.now_policy[info] = np.zeros_like(self.game.now_policy[info])
                self.game.now_policy[info][tmp_act] = 1.0
            else:
                self.game.now_policy[info] = np.ones_like(self.game.imm_regret[info]) / len(self.game.imm_regret[info])
        else:
            if self.rm_mode == 'vanilla':
                self.game.now_policy[info] = tmp_r / np.sum(tmp_r)
            elif self.rm_mode == 'eta_fix':
                self.game.now_policy[info] = (tmp_r ** self.cur_eta) / np.sum(tmp_r ** self.cur_eta)

            elif self.rm_mode == 'br':
                br_action = np.argmax(tmp_r)
                self.game.now_policy[info] = np.zeros_like(tmp_r)
                self.game.now_policy[info][br_action] = 1.0

    def all_state_regret_matching_strategy(self):
        # Regret matching in full traversal mode
        for info in self.game.now_prob.keys():
            self.regret_matching_strategy(info)
            tmp_now_prob = self.game.now_prob[info] * self.game.now_policy[info]
            tmp_now_prob = tmp_now_prob * self.ave_weight
            self.game.w_his_policy[info] += tmp_now_prob
            # The probability of reaching this information set
            self.game.now_prob[info] = 0
            if self.is_rm_plus:
                self.game.imm_regret[info][self.game.imm_regret[info] < 0] = 0.0

    def prepare_before_train(self):
        self.last_itr_node_touched = self.node_touched
        self.itr_num += 1
        self.ave_weight = self.get_ave_weight()
        self.pure_policy_num = 0
        if self.is_rm_plus:
            self.total_weight += self.cur_ave_weight
        else:
            self.total_weight += self.ave_weight

        if self.is_sampling_chance == 'no_sampling':
            # If the training is not sampled, then the entire game is played and retrained
            self.all_state_regret_matching_strategy()
        else:
            # If it is a sampled training, reset the game at the beginning of the training
            self.game.reset()

    def is_log_func(self):
        def update_log_threshold(baseline):
            while baseline >= self.log_threshold:
                if self.log_mode == 'normal':  # Equidistant storage
                    self.log_threshold += self.log_interval
                elif self.log_mode == 'exponential':  # Proportional storage
                    self.log_threshold *= self.log_interval

        now_time = time.time() - self.start_time

        if self.node_touched > self.log_state_start_num:
            # Training storage conditions
            if self.log_interval_mode == 'itr':
                if self.itr_num >= self.log_threshold:
                    self.log_model(self.itr_num, now_time)
                    self.first_log = False
                    update_log_threshold(self.itr_num)

            elif self.log_interval_mode == 'train_time':
                if now_time >= self.log_threshold:
                    self.log_model(self.itr_num, now_time)
                    self.first_log = False
                    update_log_threshold(now_time)

            elif self.log_interval_mode == 'node_touched':
                # 经过信息集个数的方式
                if self.node_touched >= self.log_threshold:
                    self.log_model(self.itr_num, now_time)
                    self.first_log = False
                    update_log_threshold(self.node_touched)

            else:
                pass

    def is_train_end_func(self):
        if self.train_mode == 'fix_itr':
            if self.itr_num >= self.total_train_constraint:
                return True
        elif self.train_mode == 'fix_node_touched':
            if self.node_touched >= self.total_train_constraint:
                return True
        elif self.train_mode == 'fix_train_time':
            if time.time() - self.start_time >= self.total_train_constraint:
                return True
        return False

    def log_epsilon(self):
        print('Start calculating epsilon')
        epsilon_start_time = time.time()
        file_list = os.listdir(self.result_file_path)
        file_list.remove('tmp.csv')
        file_list = [int(i.split('.')[0]) for i in file_list]
        file_list.sort()
        file_list = [str(i) + '.npy' for i in file_list]
        epsilon_list = []
        for i_file in file_list:
            sim_ne_policy = np.load(self.result_file_path + "/" + i_file, allow_pickle=True).item()
            epsilon, game_value = self.get_epsilon(sim_ne_policy)
            epsilon_list.append(epsilon)

        with open(self.result_file_path + '/tmp.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            rows = list(reader)
        tmp_i = 0
        with open(self.result_file_path + '/epsilon.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(rows[0])
            for row in rows[1:]:
                row[2] = epsilon_list[tmp_i]
                tmp_i += 1
                writer.writerow(row)
        print('Calculate epsilon completion and the time it takes to calculate epsilon:', time.time() - epsilon_start_time)

    def train(self):
        """
        The Training Entrance of Game Learning
        :return:
        """
        os.makedirs(self.result_file_path)

        self.start_time = time.time()

        while True:
            self.prepare_before_train()
            tmp_reward = self.walk_tree('_', np.ones(len(self.game.player_set)), 1.0)
            self.p1_total_reward += tmp_reward[0]
            # Training end conditions
            if self.is_train_end_func():
                self.log_model(self.itr_num, time.time() - self.start_time)
                self.save_model(self.itr_num)
                break

            self.is_log_func()

        print('Training has been completed：')
        print('Time spent on training:', time.time() - self.start_time)

        print(
            'itr:', self.itr_num,
            'info_num:', len(self.game.get_his_mean_policy()),
            'node_touched:', self.node_touched
        )
        if self.is_show_policy:
            print(len(self.game.get_his_mean_policy()))
            for i_keys in sorted(self.game.get_his_mean_policy().keys()):
                print(i_keys, ': ', self.game.get_his_mean_policy()[i_keys])
            print()
            print(self.game.imm_regret)
            self.all_state_regret_matching_strategy()
            for i_keys in sorted(self.game.imm_regret.keys()):
                print(i_keys, ': ', self.game.imm_regret[i_keys])
            print()
        self.start_time = time.time()
        self.log_epsilon()
