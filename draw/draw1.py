import matplotlib.pyplot as plt
import numpy as np
import os
from draw.convergence_rate import plot_once, plot_hist


def plt_normal_game_convergence(game_name, is_log=True):
    plot_once('log/convergence/CFR_' + game_name, 0, 'CFR', is_x_log=is_log, is_y_log=is_log)
    plot_once('log/convergence/SPFP_' + game_name, 1, 'BUFP(EF)', is_x_log=is_log, is_y_log=is_log)
    plot_once('log/convergence/XFP_' + game_name, 2, 'BUFP(X)', is_x_log=is_log, is_y_log=is_log)
    plt.xlabel('log10(Iterations)')
    plt.ylabel('log10(Exploitability)')
    plt.title(game_name)


def plt_normal_game_convergence1(game_name, is_log=True):
    plot_once('logGFSPSampling/br3/Train', 0, 'br3', is_x_log=is_log, is_y_log=is_log)
    plot_once('logGFSPSampling/eta03/Train', 1, 'eta03', is_x_log=is_log, is_y_log=is_log)
    plot_once('logGFSPSampling/rm+3/Train', 2, 'rm+3', is_x_log=is_log, is_y_log=is_log)
    plot_once('logGFSPSampling/rm3/Train', 3, 'rm3', is_x_log=is_log, is_y_log=is_log)
    plt.xlabel('log10(Iterations)')
    plt.ylabel('log10(Exploitability)')
    plt.title(game_name)


def plt_perfect_game_convergence_inline(game_name, logdir, is_x_log=True, is_y_log=True, y_num=3, x_num=0,
                                        log_interval_mode='node_touched'):
    file_list = os.listdir(logdir)
    file_list.sort()

    plt.ylabel('log10(Exploitability)')
    if log_interval_mode == 'node_touched':
        is_x_log = False
        is_y_log = True
        # plt.xlabel('log10(Node touched)')
        plt.xlabel('Node touched')
    elif log_interval_mode == 'train_time':
        is_x_log = False
        is_y_log = True
        plt.xlabel('Train Time')

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=is_x_log,
            is_y_log=is_y_log,
            y_num=y_num,
            x_num=x_num
        )

    plt.title(game_name)


def plt_perfect_game_convergence(game_name, is_log=True):
    plot_once('log/convergence/CFR_' + game_name, 0, 'CFR', is_x_log=is_log, is_y_log=is_log, y_num=3)
    plot_once('log/convergence/SPFP_' + game_name, 1, 'BUFP(EF)', is_x_log=is_log, is_y_log=is_log, y_num=3)
    plot_once('log/convergence/XFP_' + game_name, 2, 'BUFP(X)', is_x_log=is_log, is_y_log=is_log, y_num=3)
    plt.xlabel('log10(Iterations)')
    plt.ylabel('log10(Total Exploitability)')
    plt.title(game_name)


def plt_style_game_convergence(game_name):
    plot_once('log/style/' + game_name + '_defensive', 0, 'conservative')
    plot_once('log/style/' + game_name + '_normal', 1, 'normal')
    plot_once('log/style/' + game_name + '_offensive', 2, 'aggressive')
    plt.xlabel('log10(Iterations)')
    plt.ylabel('log10(Exploitability)')
    plt.title(game_name)


def plt_style_policy(game_name, y_num=6):
    plot_once('log/style/' + game_name + '_defensive', 0, 'conservative', is_x_log=False, is_y_log=False, y_num=y_num)
    plot_once('log/style/' + game_name + '_normal', 1, 'normal', is_x_log=False, is_y_log=False, y_num=y_num)
    plot_once('log/style/' + game_name + '_offensive', 2, 'aggressive', is_x_log=False, is_y_log=False, y_num=y_num)
    plt.xlabel('Iterations')
    plt.ylabel('Raise Prob')
    plt.title(game_name)


def plt_CFR_RC():
    plot_hist('log/RC', 'CFR', y_num=5)
    plt.xlabel('Raise Prob')
    plt.ylabel('times')


def normal_convergence():
    plt.subplot(411)
    plt_normal_game_convergence('Kuhn3')
    plt.subplot(412)
    plt_normal_game_convergence('Kuhn5')
    plt.subplot(413)
    plt_normal_game_convergence('Leduc3')
    plt.subplot(414)
    plt_normal_game_convergence('Leduc5')


def perfect_convergence():
    plt.subplot(411)
    plt_perfect_game_convergence('Kuhn3')
    plt.subplot(412)
    plt_perfect_game_convergence('Kuhn5')
    plt.subplot(413)
    plt_perfect_game_convergence('Leduc3')
    plt.subplot(414)
    plt_perfect_game_convergence('Leduc5')


def style_convergence():
    plt.subplot(221)
    plt_style_game_convergence('Kuhn3')
    plt.subplot(222)
    plt_style_policy('Kuhn3')
    plt.subplot(223)
    plt_style_game_convergence('Leduc5')
    plt.subplot(224)
    plt_style_policy('Leduc5')


def style_win_rate():
    nor = np.array([801.0, 988, 1118, 1232, 1322])
    off = np.array([785.0, 987, 1065, 1184, 1281])
    defen = np.array([799.0, 1050, 1167, 1301, 1430])
    x = [10, 15, 20, 25, 30]
    nor /= 2000
    off /= 2000
    defen /= 2000
    plt.plot(x, nor, marker='o', label='normal')
    plt.plot(x, off, marker='o', label='aggressive')
    plt.plot(x, defen, marker='o', label='conservative')
    plt.xlabel('Chips')
    plt.ylabel('Win Rate')


def plt_normal_game_convergence_CG(is_log=True):
    plot_once('log/convergence/CFR_CG', 0, 'CFR', is_x_log=is_log, is_y_log=is_log)
    plot_once('log/convergence/SPFP_CG', 1, 'BUFP(EF)', is_x_log=is_log, is_y_log=is_log)
    plt.xlabel('log10(Iterations)')
    plt.ylabel('log10(Exploitability)')
    plt.title('Toy matrix-like game')


if __name__ == '__main__':
    plt.figure(figsize=(32, 10), dpi=60)
    # plt.figure(dpi=320)

    # plt_normal_game_convergence_CG()
    # plt_CFR_RC()
    # perfect_convergence()
    # style_convergence()
    # normal_convergence()
    # style_win_rate()
    # plt_perfect_game_convergence('Kuhn5')
    # plt_style_policy('')

    # plt.tight_layout()
    # # plt.axis()
    # plt.legend(edgecolor='red')
    # plt.show()
    #
    plt.subplot(1, 2, 1)
    plt_perfect_game_convergence_inline(
        '15_LeakyLeduc',
        '/home/root523/workspace/ft/new_gwpfefg/logGFSPSampling/important_15_LeakyLeduc',
        is_x_log=False,
        x_num=4,
        y_num=2,
        log_interval_mode='node_touched'
    )
    plt.subplot(1, 2, 2)
    plt_perfect_game_convergence_inline(
        '15_LeakyLeduc',
        '/home/root523/workspace/ft/new_gwpfefg/logGFSPSampling/important_15_LeakyLeduc',
        is_x_log=False,
        x_num=1,
        y_num=2,
        log_interval_mode='train_time'
    )

    plt.tight_layout()  # Automatically adjust the spacing between subgraphs, axes, and titles to make the image more compact and aesthetically pleasing.
    # plt.axis()
    plt.legend(edgecolor='red')  # Set the position and color of the legend
    # plt.savefig(logdir + '/' + total_exp_name + '/pic.png')
    plt.show()
