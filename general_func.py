import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

plt.rcParams['font.sans-serif']=['STSong'] # 用来正常显示中文标签
plt.rcParams['font.size']=14
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号


def draw(time_slot, config):
    ds = config.distance
    numS = config.n_stations
    time_horizon = config.time_zone

    xcoodi = [0, time_horizon]
    plt.xticks(xcoodi)
    plt.xlabel('time(min)')
    ##    time_horizon = 140
    # print(ds)
    ycoodi = []
    d = 0
    for i in ds[::-1]:
        d += i
        ycoodi.append(d)

    ycoodi.insert(0, 0)
    ycoodi = [round(ycoodi[i], 1) for i in range(len(ycoodi))]

    maxy = ycoodi[-1]
    plt.xlim((0, time_horizon))
    plt.ylim((0, maxy))

    yticks = np.arange(numS)
    yticks = yticks[::-1]

    # plt.yticks(ycoodi)
    plt.yticks(ycoodi, labels=yticks[-numS:])
    # plt.yticks(ycoodi)
    for i in xcoodi:
        x = [i, i]
        y = [0, maxy]
        plt.plot(x, y, color='gray')

    for i in ycoodi:
        x = [0, time_horizon]
        y = [i, i]
        plt.plot(x, y, color='gray')

    revs_y = ycoodi[::-1]
    for i in range(len(time_slot) // 2):
        departure = time_slot[2 * i]
        arrive = time_slot[2 * i + 1]
        for j in range(len(departure)):
            d = departure[j]
            a = arrive[j]
            x = [d, a]
            y = [revs_y[i], revs_y[i + 1]]
            if j < len(departure) / 2:
                plt.plot(x, y, color='r', linewidth=1)
            else:
                plt.plot(x, y, color='b', linewidth=1)

    plt.ylabel('Space')

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.show()


import pickle
# import xlrd
import numpy as np


def write_file(file_name, file):
    file_name = file_name + '.pkl'
    with open(file_name, 'wb') as fp:
        pickle.dump(file, fp)


def read_file(file_name):
    file_name = file_name + '.pkl'
    with open(file_name, 'rb') as fp:
        return pickle.load(fp)