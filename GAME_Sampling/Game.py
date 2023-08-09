import numpy as np
import copy


class Game(object):
    def __init__(self, config: dict):
        self.game_name = config.get('game_name', 'Please provide a game name！')

        self.prior_state_num = config.get('prior_state_num', 3)
        self.player_set = config.get('player_set', ['player1', 'player2'])
        self.info_set_list = {}
        self.pri_feat = {}
        for p in self.player_set:
            self.info_set_list[p] = []
            self.pri_feat[p] = '_'
        self.terminal_list = []
        self.imm_regret = {}
        self.now_policy = {}
        self.now_prob = {}
        self.w_his_policy = {}

        self.itr = 0
        self.game_train_mode = 'vanilla'

    def generate_new_info_set(self, tmp_info_set, tmp_now_player, next_action_len):
        self.imm_regret[tmp_info_set] = np.zeros(next_action_len)
        if self.game_train_mode == 'PCFR':
            self.now_policy[tmp_info_set] = np.random.randint(next_action_len)
        else:
            self.now_policy[tmp_info_set] = np.random.random(next_action_len)
            self.now_policy[tmp_info_set] = self.now_policy[tmp_info_set] / np.sum(self.now_policy[tmp_info_set])

        self.w_his_policy[tmp_info_set] = np.zeros(next_action_len)
        self.info_set_list[tmp_now_player].append(tmp_info_set)
        self.now_prob[tmp_info_set] = 0

    def reset(self):
        """
        Reset game
        :return:
        """
        pass

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        """
        According to the historical process, get the current round of players
        :param his_feat:
        :return: ['c','player1','player2',...]
        """
        pass

    def judge(self, his_feat) -> np.ndarray:
        """
        Get utility based on historical processes
        :param his_feat:
        :return:
        """
        pass

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        """
        Get legal action sets based on historical processes
        :param his_feat:
        :return: If the action set is not returned by the terminal node, otherwise an empty set is returned
        """
        pass

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        """
        Get the probability of opportunity nodes based on historical processes
        :param his_feat:
        :return: Return various possible probabilities
        """
        pass

    def get_deterministic_chance_action(self, his_feat: str) -> str:
        """
        In MCCFR, sampling can also be done for opportunity nodes, so an action determined by the opportunity node can be generated based on historical processes.
        :param his_feat:
        :return: Deterministic actions
        """
        pass

    def get_sum_imm_regret(self):
        """
        Get historical overall regret value
        :return:
        """
        tmp_regret = copy.deepcopy(self.imm_regret)
        imm_regret_sum_per_player = np.zeros(len(self.player_set))

        for i_player in range(len(self.player_set)):
            for tmp_info in self.info_set_list[self.player_set[i_player]]:
                imm_regret_sum_per_player[i_player] += np.max(np.max(tmp_regret[tmp_info]), 0)  # 如果这是一个劣解，那么就要和0比较

        return imm_regret_sum_per_player

    def get_his_mean_policy(self) -> dict:
        """
        Get the average strategy of history
        :return:
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
        Get the next historical process based on the current historical process and actions
        :param his_feat:
        :param now_action:
        :return:
        """
        return his_feat + now_action

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        """
        Get the current process observed by players through historical processes
        :param his_feat:
        :return:
        """
        return his_feat

    def get_info_set(self, player_id, his_feat):
        """
        Get the current information set
        :param player_id:
        :param his_feat:
        :return:
        """
        return self.pri_feat[player_id] + self.get_pub_feat_from_his_feat(his_feat)

    def game_flow(self, policy_list: dict, is_show=False):
        """
        Simulation Battle
        :param is_show:
        :param policy_list:
        :return:
        """

        def game_sim(his_feat):
            now_player = self.get_now_player_from_his_feat(his_feat)
            if now_player == 'c':
                if is_show:
                    print('chance', now_player, '\'s round')
                tmp_act = self.get_deterministic_chance_action(his_feat)
                return game_sim(his_feat + tmp_act)
            else:
                now_info = self.get_info_set(now_player, his_feat)
                tmp_legal_actions = self.get_legal_action_list_from_his_feat(his_feat)
                if tmp_legal_actions:
                    if now_player in policy_list.keys():
                        if is_show:
                            print('AI Player', now_player, '\'s round')
                        if now_info in policy_list[now_player].keys():
                            tmp_act = np.random.choice(tmp_legal_actions, p=policy_list[now_player][now_info])
                        else:
                            tmp_act = np.random.choice(tmp_legal_actions)
                        return game_sim(his_feat + tmp_act)
                    else:
                        print('Human player \'s turn')
                        print('The current information set is：', now_info)
                        if now_info[0] != '_' and self.game_name == 'ThreeCardPoker':
                            print('hand：')
                            if now_player == 'player1':
                                print(self.poker[:3])
                            else:
                                print(self.poker[3:6])

                        correct_input = False
                        while not correct_input:
                            try:
                                print('The optional actions are：', tmp_legal_actions, 'please input(1,2,3...)')
                                human_act_index = input()
                                human_act = tmp_legal_actions[int(human_act_index) - 1]
                                correct_input = True
                            except:
                                print('Input error,please input again')

                        return game_sim(his_feat + human_act)
                else:
                    result = self.judge(his_feat)
                    if is_show:
                        print('Game over：')
                        print('The entire process of the game：', his_feat)
                        print('Game score：', result)
                        print('Players\'s Hand：', self.poker[:6])
                    return result

        return game_sim('_')
