import numpy as np
import copy


class Game(object):
    def __init__(self, config: dict):
        self.game_name = config.get('game_name')
        self.prior_state_num = config.get('prior_state_num', 3)
        self.player_num = config.get('player_num', 2)
        self.player_set = ['player' + str(i + 1) for i in range(self.player_num)]
        self.info_set_list = {}
        self.pri_feat = {}
        for p in self.player_set:
            self.info_set_list[p] = []
            self.pri_feat[p] = '_'
        self.terminal_list = []
        self.his_regret = {}
        self.now_policy = {}
        self.now_prob = {}
        self.w_his_policy = {}

        self.itr = 0
        self.game_train_mode = 'vanilla'

    def generate_new_info_set(self, info_set_name, now_player, next_action_len):
        """
        generate new info set
        """
        self.his_regret[info_set_name] = np.zeros(next_action_len)
        if self.game_train_mode == 'CFVFP':
            self.now_policy[info_set_name] = np.random.randint(next_action_len)
        else:
            self.now_policy[info_set_name] = np.random.random(next_action_len)
            self.now_policy[info_set_name] = self.now_policy[info_set_name] / np.sum(self.now_policy[info_set_name])

        self.w_his_policy[info_set_name] = np.zeros(next_action_len)
        self.info_set_list[now_player].append(info_set_name)
        self.now_prob[info_set_name] = 0

    def reset(self):
        pass

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        pass

    def judge(self, his_feat) -> np.ndarray:
        """
        return the reward of each player
        """
        pass

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        pass

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        """
        return the chance probability of each action in the chance node
        """
        pass

    def get_deterministic_chance_action(self, his_feat: str) -> str:
        """
        return the deterministic chance action in the chance node
        """
        pass

    def get_sum_his_regret(self):

        tmp_regret = copy.deepcopy(self.his_regret)
        his_regret_sum_per_player = np.zeros(len(self.player_set))

        for i_player in range(len(self.player_set)):
            for tmp_info in self.info_set_list[self.player_set[i_player]]:
                his_regret_sum_per_player[i_player] += np.max(np.max(tmp_regret[tmp_info]), 0)  # 如果这是一个劣解，那么就要和0比较

        return his_regret_sum_per_player

    def get_his_mean_policy(self) -> dict:
        """
        return the mean policy of each info set
        """
        tmp_his_policy = copy.deepcopy(self.w_his_policy)
        for i_key in tmp_his_policy.keys():
            if np.sum(tmp_his_policy[i_key]) == 0:
                tmp_his_policy[i_key] = np.ones_like(tmp_his_policy[i_key]) / len(tmp_his_policy[i_key])
            else:
                tmp_his_policy[i_key] = tmp_his_policy[i_key] / np.sum(tmp_his_policy[i_key])
        return tmp_his_policy

    def get_next_his_feat(self, his_feat, now_action) -> str:
        """
        return the next history feature after taking the now_action on the self.
        """
        return his_feat + now_action

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        return his_feat

    def get_info_set(self, player_id, his_feat):
        return self.pri_feat[player_id] + self.get_pub_feat_from_his_feat(his_feat)
