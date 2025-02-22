import os
import numpy as np
import matplotlib.pyplot as plt
plot_color = [
    [
        (57 / 255, 81 / 255, 162 / 255),  # deep
        (202 / 255, 232 / 255, 242 / 255),  # middle
        (114 / 255, 170 / 255, 207 / 255),  # light

    ],
    [
        (168 / 255, 3 / 255, 38 / 255),
        (253 / 255, 185 / 255, 107 / 255),
        (236 / 255, 93 / 255, 59 / 255),
    ],
    [
        (0 / 255, 128 / 255, 51 / 255),
        (202 / 255, 222 / 255, 114 / 255),
        (226 / 255, 236 / 255, 179 / 255)
    ],
    [
        (128 / 255, 0 / 255, 128 / 255),
        (204 / 255, 153 / 255, 255 / 255),
        (128 / 255, 128 / 255, 128 / 255)
    ],
    [
        (255 / 255, 215 / 255, 0 / 255),  # yellow
        (255 / 255, 239 / 255, 213 / 255),
        (255 / 255, 250 / 255, 240 / 255),
    ],
    [
        (207 / 255, 145 / 255, 151 / 255),  # pink
        (231 / 255, 208 / 255, 211 / 255),
        (245 / 255, 239 / 255, 238 / 255),
    ],
    [
        (255 / 255, 165 / 255, 0 / 255),  # orange
        (255 / 255, 192 / 255, 128 / 255),
        (255 / 255, 224 / 255, 192 / 255),
    ],
    [
        (184 / 255, 146 / 255, 106 / 255),  # brown
        (210 / 255, 191 / 255, 166 / 255),
        (239 / 255, 237 / 255, 231 / 255),
    ],
    [
        (63 / 255, 55 / 255, 54 / 255),  # black
        (126 / 255, 127 / 255, 122 / 255),
        (234 / 255, 230 / 255, 223 / 255),
    ],
]

plot_marker = ['s', '^', '*', 'o', 'D', 'x', '+', '<', '>', 'P', 'p']


def get_file_name_list(path: str) -> list:
    file_list = os.listdir(path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    csv_file = []
    for i_file in file_list:
        if i_file[-2:] == 'WS':
            continue
        csv_file.append(path + '/' + i_file + '/epsilon.csv')
    return csv_file


def get_result(file_path: str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=',', skiprows=2)

    return data


def plot_once(path, num, ex_name, is_x_log=True, is_y_log=True, y_label_index=3, x_label_index=0):
    csv_file_list = get_file_name_list(path)
    _10_num = int(0.1 * len(csv_file_list))
    one_trail_data = get_result(csv_file_list[0])

    if is_x_log:
        x_data = np.log10(one_trail_data[:, x_label_index])
    else:
        x_data = one_trail_data[:, x_label_index]

    tmp_min_x = one_trail_data.shape[0]

    mean_y_data = one_trail_data[:, y_label_index]
    y_data_matrix = np.zeros((len(csv_file_list), tmp_min_x))
    y_data_matrix[0, :] = mean_y_data

    for i in range(1, len(csv_file_list)):
        one_trail_data = get_result(csv_file_list[i])
        now_min_x = one_trail_data.shape[0]
        if now_min_x < tmp_min_x:
            tmp_min_x = now_min_x
            y_data_matrix = y_data_matrix[:, -tmp_min_x:]
            mean_y_data = mean_y_data[-tmp_min_x:]
            x_data = x_data[-tmp_min_x:]
        one_trail_data = one_trail_data[-tmp_min_x:, :]

        mean_y_data += one_trail_data[:, y_label_index]
        y_data_matrix[i, :] = one_trail_data[:, y_label_index]
        if is_y_log:
            plt.scatter(x_data, np.log10(one_trail_data[:, y_label_index]), s=1, marker=plot_marker[num],
                        color=plot_color[num][1], alpha=0.4)
        else:
            plt.scatter(x_data, one_trail_data[:, y_label_index], s=1, marker=plot_marker[num],
                        color=plot_color[num][1], alpha=0.4)
    y_data_matrix.sort(axis=0)

    mean_y_data = mean_y_data / len(csv_file_list)
    if is_y_log:
        mean_y_data = np.log10(mean_y_data)
        y_data_matrix = np.log10(y_data_matrix)

    plt.plot(x_data, mean_y_data, marker=plot_marker[num], markersize=10, c=plot_color[num][0], lw=2, label=ex_name)

    plt.fill_between(x_data, y_data_matrix[_10_num, :], y_data_matrix[-_10_num - 1, :], color=plot_color[num][2],
                     alpha=0.3)
    plt.tick_params(axis='both', labelsize=20)


def plt_perfect_game_convergence_inline(game_name, logdir, is_x_log=True, is_y_log=True, y_label_index=3,
                                        x_label_index=0,
                                        x_label_name='X', y_label_name='Y'):
    file_list = os.listdir(logdir)
    file_list.sort()
    plt.xticks(fontproperties='Times New Roman', size=18)
    plt.yticks(fontproperties='Times New Roman', size=18)

    plt.ylabel(y_label_name, fontproperties='Times New Roman', size=18)
    plt.xlabel(x_label_name, fontproperties='Times New Roman', size=18)

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=is_x_log,
            is_y_log=is_y_log,
            y_label_index=y_label_index,
            x_label_index=x_label_index
        )

    plt.title(game_name, fontproperties='Times New Roman', size=24)
    plt.legend(edgecolor='red', fontsize=18)
