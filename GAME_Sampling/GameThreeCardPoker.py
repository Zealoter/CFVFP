from GAME_Sampling.Game import Game
import numpy as np
from itertools import permutations


def GetTypeAndChance(prior_state_num):
    card = []
    for i in range(prior_state_num):
        card.append('♥' + str(i + 2))
        card.append('♠' + str(i + 2))
        card.append('♣' + str(i + 2))
        card.append('♦' + str(i + 2))

    action_frequency_dict = {}
    for hand_cards in permutations(card, 6):
        card1 = ''.join(hand_cards[:3])
        card2 = ''.join(hand_cards[3:])
        a = get_card_power(card1[1], card1[3], card1[5], card1[0], card1[2], card1[4])
        b = get_card_power(card2[1], card2[3], card2[5], card2[0], card2[2], card2[4])
        tmp_act = a + '_' + b + '_'
        if tmp_act in action_frequency_dict.keys():
            action_frequency_dict[tmp_act] += 1
        else:
            action_frequency_dict[tmp_act] = 1
    start_c_action_list = list(action_frequency_dict.keys())

    list1 = []
    for i_c_act in start_c_action_list:
        list1.append(action_frequency_dict[i_c_act])
    now_prob = np.array(list1)
    now_prob = now_prob / np.sum(now_prob)

    return now_prob, start_c_action_list


def get_card_power(num1, num2, num3, color1, color2, color3):  # General Method for Obtaining Card Strength
    cards_list = [int(num1), int(num2), int(num3)]
    suites_list = [color1, color2, color3]
    cards_list.sort()
    result = 0
    straight = 0
    if len(set(cards_list)) == 3:
        if cards_list[0] + 2 == cards_list[2]:
            straight = 1
    # 6-Three cards of the same numerical size 5-straight flush  4-flush  3-straight  2-pair  1-high card
    if len(set(suites_list)) == 1:
        if straight == 1:
            result = cards_list[2]
            result = 5 << 20 | result
        else:
            for i in range(3):
                result = result | cards_list[i] << 4 * i
            result = 4 << 20 | result
    elif straight == 1:
        result = cards_list[2]
        result = 3 << 20 | result
    elif len(set(cards_list)) == 1:
        result = cards_list[0]
        result = 6 << 20 | result
    else:
        if cards_list[0] == cards_list[1]:
            result = result | cards_list[0] << 12
            result = result | cards_list[2] << 8
            result = 2 << 20 | result
        elif cards_list[1] == cards_list[2]:
            result = result | cards_list[1] << 12
            result = result | cards_list[0] << 8
            result = 2 << 20 | result
        elif cards_list[0] == cards_list[2]:
            result = result | cards_list[2] << 12
            result = result | cards_list[1] << 8
            result = 2 << 20 | result
        else:
            for i in range(3):
                result = result | cards_list[i] << 4 * i
            result = 1 << 20 | result
    return str(result)


class ThreeCardPoker(Game):
    def __init__(self, config):
        super().__init__(config)

        self.poker = []
        for i in range(self.prior_state_num):  # prior_state_num is the number of cards for each flower color
            self.poker.append('♥' + str(i + 2))
            self.poker.append('♠' + str(i + 2))
            self.poker.append('♣' + str(i + 2))
            self.poker.append('♦' + str(i + 2))

        np.random.shuffle(self.poker)
        self.get_card_strength()
        self.chance_list, self.type_list = GetTypeAndChance(self.prior_state_num)
        # self.chance_list = GetTypeAndChance(self.prior_state_num, 'chance')

    def get_card_strength(self):
        self.pri_feat['player1'] = get_card_power(
            self.poker[0][1],
            self.poker[1][1],
            self.poker[2][1],
            self.poker[0][0],
            self.poker[1][0],
            self.poker[2][0],
        )

        self.pri_feat['player2'] = get_card_power(
            self.poker[3][1],
            self.poker[4][1],
            self.poker[5][1],
            self.poker[3][0],
            self.poker[4][0],
            self.poker[5][0],
        )

    def reset(self):
        np.random.shuffle(self.poker)
        self.get_card_strength()

    def get_now_player_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return 'c'
        elif len(his_feat.split('_')[-1]) % 4 <= 1:
            return 'player1'
        else:
            return 'player2'

    def get_legal_action_list_from_his_feat(self, his_feat: str) -> list:
        if his_feat == '_':
            return self.type_list
        tmp_h = his_feat.split('_')[-1]
        is_shown_flag = False
        if len(tmp_h) > 0 and (tmp_h[-1] == 'T' or tmp_h[-1] == 'F' or len(tmp_h) >= 12):  # 终止条件
            return []
        elif len(tmp_h) % 2 == 0:
            if 'S' in tmp_h[-4::-4]:
                is_shown_flag = True
            else:
                return ['S', 'N']

        if 'R' in tmp_h:
            action_list = ['F', 'C', 'T']
        elif 'r' in tmp_h:
            action_list = ['F', 'C', 'R', 'T']
        else:
            action_list = ['F', 'C', 'r', 'R', 'T']

        if len(tmp_h) <= 4:
            action_list.remove('T')
        if is_shown_flag:
            action_list = ['S' + i_act for i_act in action_list]

        return action_list

    def judge(self, his_feat):
        money_p1 = 1.0
        money_p2 = 1.0
        card1 = his_feat.split('_')[1]
        card2 = his_feat.split('_')[2]
        tmp_h = ''.join(his_feat.split('_')[3:])
        now_pot = 1
        is_p1_shown = 1
        is_p2_shown = 1

        for i in range(len(tmp_h) // 2):
            if tmp_h[i * 2] == 'S':
                if i % 2 == 0:
                    is_p1_shown = 2
                else:
                    is_p2_shown = 2
            if tmp_h[i * 2 + 1] == 'F':
                if i % 2 == 0:
                    return np.array([-money_p1, money_p1])
                else:
                    return np.array([money_p2, -money_p2])
            elif tmp_h[i * 2 + 1] == 'C':
                pass
            elif tmp_h[i * 2 + 1] == 'r':
                now_pot = 2
            elif tmp_h[i * 2 + 1] == 'R':
                now_pot = 4
            elif tmp_h[i * 2 + 1] == 'T':
                if i % 2 == 0:
                    money_p1 += (is_p1_shown * now_pot * 2)
                else:
                    money_p2 += (is_p2_shown * now_pot * 2)
                break
            if i % 2 == 0:
                money_p1 += (is_p1_shown * now_pot)
            else:
                money_p2 += (is_p2_shown * now_pot)

        if int(card1) > int(card2):
            return np.array([money_p2, -money_p2])
        elif int(card1) == int(card2):
            mean_money = (money_p1 + money_p2) / 2
            return np.array([mean_money - money_p1, mean_money - money_p2])
        else:
            return np.array([-money_p1, money_p1])

    def get_chance_prob(self, his_feat: str) -> np.ndarray:
        return self.chance_list

    def get_deterministic_chance_action(self, his_feat: str) -> str:
        return self.pri_feat['player1'] + '_' + self.pri_feat['player2'] + '_'

    def get_info_set(self, player_id, his_feat):
        his_feat_list = his_feat.split('_')
        self_feat = ''
        if player_id == 'player1':
            is_shown = his_feat_list[3][::4]
            if 'S' in is_shown:
                self_feat = his_feat_list[1]
        else:
            is_shown = his_feat_list[3][2::4]
            if 'S' in is_shown:
                self_feat = his_feat_list[2]

        return self_feat + self.get_pub_feat_from_his_feat(his_feat)

    def get_pub_feat_from_his_feat(self, his_feat: str) -> str:
        if his_feat == '_':
            return his_feat
        else:
            pub_feat_list = his_feat.split('_')
            ob_his_fear = '_'.join(pub_feat_list[3:])
            return '_' + ob_his_fear
