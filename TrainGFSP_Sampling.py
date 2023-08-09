"""
# @Author: JuQi
# @Time  : 2023/3/6 19:08
# @E-mail: 18672750887@163.com
"""
import copy
import time
import numpy as np
from GAME_Sampling.GameKuhn import Kuhn
from GAME_Sampling.GameLeduc import Leduc
from GAME_Sampling.GameLeduc5Pot import Leduc5Pot
from GAME_Sampling.GameLeduc3Pot import Leduc3Pot
from GAME_Sampling.GameGoofspiel import Goofspiel
from GAME_Sampling.GameKuhnNPot import KuhnNPot
from GAME_Sampling.GamePrincessAndMonster import PrincessAndMonster as PAM
from GAME_Sampling.GameThreeCardPoker import ThreeCardPoker
from GAME_Sampling.GameMatrix import Matrix
from GFSP_Sampling.PCFR import PCFRSolver
from GFSP_Sampling.GFSP import GFSPSamplingSolver
from GFSP_Sampling.WarmStartCFR import WSCFRSolver

from CONFIG import test_sampling_train_config

from draw.convergence_rate import plt_perfect_game_convergence_inline
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc


def train_sec(tmp_train_config):
    op_env = tmp_train_config.get('op_env', 'GFSP')
    if op_env == 'PCFR':
        tmp = PCFRSolver(tmp_train_config)
    elif op_env == 'GFSP':
        tmp = GFSPSamplingSolver(tmp_train_config)
    elif op_env == 'WS':
        tmp = WSCFRSolver(tmp_train_config)
    else:
        return
    tmp.train()
    del tmp
    gc.collect()
    return


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format}, suppress=True)
    logdir = 'logGFSPSampling'
    game_name = 'Kuhn'
    is_show_policy = False
    prior_state_num = 3
    y_pot = 3
    z_len = 3

    game_config = {
        'game_name'      : game_name,
        'prior_state_num': prior_state_num,
        'y_pot'          : y_pot,
        'z_len'          : z_len,
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
    elif game_name == 'ThreeCardPoker':
        game_class = ThreeCardPoker(game_config)
    elif game_name == 'Matrix':
        game_class = Matrix(game_config)

    # train_mode = 'fix_itr'
    train_mode = 'fix_node_touched'
    # train_mode = 'fix_train_time'
    # log_interval_mode = 'itr'
    log_interval_mode = 'node_touched'
    # log_interval_mode = 'train_time'
    # log_mode = 'normal'
    log_mode = 'exponential'

    total_train_constraint = 100000
    log_interval = 2
    nun_of_train_repetitions = 1
    n_jobs = 1  # Parallel run settings

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

    if log_interval_mode == 'itr':
        plot_x_index = 0
    elif log_interval_mode == 'node_touched':
        plot_x_index = 4
    elif log_interval_mode == 'train_time':
        plot_x_index = 1
    else:
        print('plot_x_index is incorrect')
        plot_x_index = 0

    plt.subplot(1, 2, 1)
    plt_perfect_game_convergence_inline(
        fig_title,
        logdir + '/' + total_exp_name,
        is_x_log=True,
        x_num=4,
        y_num=2,
        log_interval_mode='node_touched'
    )
    plt.subplot(1, 2, 2)
    plt_perfect_game_convergence_inline(
        fig_title,
        logdir + '/' + total_exp_name,
        is_x_log=False,
        x_num=1,
        y_num=2,
        log_interval_mode='train_time'
    )

    plt.tight_layout()
    # plt.axis()
    plt.legend(edgecolor='red')
    plt.savefig(logdir + '/' + total_exp_name + '/pic.png')
    plt.show()
