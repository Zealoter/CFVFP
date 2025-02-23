import copy
import time
import numpy as np
from GAME.GameKuhn import Kuhn
from GAME.GameLeduc import Leduc
from GAME.GameLeduc5Pot import Leduc5Pot
from GAME.GameLeduc3Pot import Leduc3Pot
from GAME.GameGoofspiel import Goofspiel
from GAME.GameKuhnNPot import KuhnNPot
from GAME.GamePrincessAndMonster import PrincessAndMonster as PAM
from Solver.CFVFP import CFVFPSolver
from Solver.CFR import CFRSolver
from CONFIG import test_sampling_train_config

import draw.convergence_rate
from draw.convergence_rate import plt_perfect_game_convergence_inline
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc


def train_sec(tmp_train_config):
    op_env = tmp_train_config.get('op_env', 'CFR')
    if op_env == 'CFVFP':
        tmp = CFVFPSolver(tmp_train_config)
    elif op_env == 'CFR':
        tmp = CFRSolver(tmp_train_config)
    else:
        return
    tmp.train()
    del tmp
    gc.collect()
    return


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    logdir = 'logCFRSampling'
    game_name = 'Leduc'
    is_show_policy = False
    prior_state_num = 15
    y_pot = 1
    z_len = 1

    game_config = {
        'game_name'      : game_name,
        'prior_state_num': prior_state_num,
        'y_pot'          : y_pot,
        'z_len'          : z_len,
        'player_num'     : 2
    }

    if game_name == 'Leduc':
        game_class = Leduc(game_config)
    elif game_name == 'Kuhn':
        game_class = Kuhn(game_config)
    elif game_name == 'Goofspiel':
        game_class = Goofspiel(game_config)
    elif game_name == 'Leduc3Pot':
        game_class = Leduc3Pot(game_config)
    elif game_name == 'Leduc5Pot':
        game_class = Leduc5Pot(game_config)
    elif game_name == 'KuhnNPot':
        game_class = KuhnNPot(game_config)
    elif game_name == 'PAM':
        game_class = PAM(game_config)


    # train_mode = 'fix_itr'
    train_mode = 'fix_node_touched'
    # train_mode = 'fix_train_time'

    # log_interval_mode = 'itr'
    log_interval_mode = 'node_touched'
    # log_interval_mode = 'train_time'

    # log_mode = 'normal'
    log_mode = 'exponential'

    total_train_constraint = 10000000
    log_interval = 1.5
    nun_of_train_repetitions = 9
    n_jobs = 9

    total_exp_name = str(prior_state_num) + '_' + game_name + '_' + time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                  time.localtime(time.time()))

    for key in test_sampling_train_config.keys():
        start = time.time()
        print(key)
        parallel_train_config_list = []
        for i_train_repetition in range(nun_of_train_repetitions):
            train_config = copy.deepcopy(test_sampling_train_config[key])

            train_config['game'] = copy.deepcopy(game_class)
            train_config['game_info'] = key
            train_config['train_mode'] = train_mode
            train_config['log_interval_mode'] = log_interval_mode
            train_config['log_mode'] = log_mode
            train_config['is_show_policy'] = is_show_policy

            train_config['total_exp_name'] = total_exp_name
            train_config['total_train_constraint'] = total_train_constraint
            train_config['log_interval'] = log_interval

            train_config['No.'] = i_train_repetition

            parallel_train_config_list.append(train_config)

        ans_list = Parallel(n_jobs=n_jobs)(
            delayed(train_sec)(i_train_config) for i_train_config in parallel_train_config_list
        )

        end = time.time()
        print(end - start)

    plt.figure(figsize=(32, 10), dpi=60)

    if game_name == 'KuhnNPot':
        fig_title = str(prior_state_num) + 'C' + str(y_pot) + 'P' + str(z_len) + 'L_Kuhn'
    else:
        fig_title = str(prior_state_num) + '_' + game_name

    plt.subplot(1, 2, 1)
    plt_perfect_game_convergence_inline(
        fig_title,
        logdir + '/' + total_exp_name,
        is_x_log=False,
        is_y_log=True,
        x_label_index=4,
        y_label_index=2,
        x_label_name='node touched',
        y_label_name='epsilon'
    )
    plt.subplot(1, 2, 2)
    plt_perfect_game_convergence_inline(
        fig_title,
        logdir + '/' + total_exp_name,
        is_x_log=False,
        is_y_log=True,
        x_label_index=1,
        y_label_index=2,
        x_label_name='time(ms)',
        y_label_name='epsilon'
    )

    plt.savefig(logdir + '/' + total_exp_name + '/pic.png')
    # plt.show()
