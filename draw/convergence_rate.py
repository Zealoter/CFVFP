import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plot_color = [
    [
        (57 / 255, 81 / 255, 162 / 255),
        (202 / 255, 232 / 255, 242 / 255),
        (114 / 255, 170 / 255, 207 / 255),

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
        (255 / 255, 215 / 255, 0 / 255),
        (255 / 255, 239 / 255, 213 / 255),
        (255 / 255, 250 / 255, 240 / 255),
    ],
    [
        (0 / 255, 0 / 255, 0 / 255),
        (64 / 255, 64 / 255, 64 / 255),
        (128 / 255, 128 / 255, 128 / 255),
    ],
    [
        (255 / 255, 165 / 255, 0 / 255),
        (255 / 255, 192 / 255, 128 / 255),
        (255 / 255, 224 / 255, 192 / 255),
    ],
]


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
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data


def plot_once(path, num, ex_name, is_x_log=True, is_y_log=True, y_num=3, x_num=0):
    # If the object is time: x_ Num=1, if the object is itr: x_ Num=0
    csv_file_list = get_file_name_list(path)  # Obtain all epsilon files in the sub file
    _10_num = int(0.1 * len(csv_file_list))
    tmp_data = get_result(csv_file_list[0])

    if is_x_log:
        tmp_x = np.log10(tmp_data[:, x_num])  # training times
    else:
        tmp_x = tmp_data[:, x_num]
    tmp_min_x = tmp_data.shape[0]
    tmp_y = tmp_data[:, y_num]

    y_matrix = np.zeros((len(csv_file_list), tmp_min_x))

    y_matrix[0, :] = tmp_data[:, y_num]
    for i in range(1, len(csv_file_list)):
        tmp_data = get_result(csv_file_list[i])
        now_min_x = tmp_data.shape[0]
        if now_min_x < tmp_min_x:
            tmp_min_x = now_min_x
            y_matrix = y_matrix[:, -tmp_min_x:]
            tmp_y = tmp_y[-tmp_min_x:]
            tmp_x = tmp_x[-tmp_min_x:]
        tmp_data = tmp_data[-tmp_min_x:, :]

        tmp_y += tmp_data[:, y_num]
        y_matrix[i, :] = tmp_data[:, y_num]
        if is_y_log:
            plt.scatter(tmp_x, np.log10(tmp_data[:, y_num]), s=1, color=plot_color[num][1], alpha=0.7)
        else:
            plt.scatter(tmp_x, tmp_data[:, y_num], s=1, color=plot_color[num][1], alpha=0.3)
    y_matrix.sort(axis=0)
    tmp_y = tmp_y / len(csv_file_list)
    if is_y_log:
        tmp_y = np.log10(tmp_y)
        y_matrix = np.log10(y_matrix)

    plt.plot(tmp_x, tmp_y, c=plot_color[num][0], lw=2, label=ex_name)
    plt.fill_between(tmp_x, y_matrix[_10_num, :], y_matrix[-_10_num - 1, :], color=plot_color[num][2], alpha=0.5)


def plot_hist(path, ex_name, y_num=2):
    csv_file_list = get_file_name_list(path)
    tmp_data = get_result(csv_file_list[0])

    tmp_x = [tmp_data[y_num]]

    for i in range(1, len(csv_file_list)):
        tmp_data = get_result(csv_file_list[i])
        tmp_x.append(tmp_data[y_num])
    sns.histplot(tmp_x, bins=20, color=plot_color[0][0], kde=True)
