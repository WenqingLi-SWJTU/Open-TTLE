import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

plt.rcParams['font.sans-serif']=['STSong'] # 用来正常显示中文标签
plt.rcParams['font.size']=14
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号


def draw(distance, num_S, time_slot, time_horizon):
    ds = distance
    numS = num_S
    # font = {'family' : 'Times New Roman', 'size': 14}
    # if case_id == 1:
    #     time_horizon = 57
    #     xcoodi = [0, time_horizon]
    #     plt.xticks(xcoodi)
    #     # plt.xlabel('Time (min)', font)
    #     plt.xlabel('时间(分钟)')
    # elif case_id == 2:
    #     time_horizon = 140
    #     xcoodi = [0, time_horizon]
    #     plt.xticks(xcoodi)
    #     # plt.xlabel('Time (min)', font)
    #     plt.xlabel('时间(分钟)')
    # else:
    #     time_horizon = 1440
    #     xcoodi = list(range(0, time_horizon+60, 60))
    #     xticks = ['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
    #               '23', '24', '1', '2', '3', '4', '5', '6']
    #     plt.xticks(xcoodi, labels=xticks)
    #     # plt.xlabel('Time (h)', font)
    #     plt.xlabel('时间(小时)')
    xcoodi = [0, time_horizon]
    plt.xticks(xcoodi)
    plt.xlabel('时间(分钟)')
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

    yticks = ['RA', 'PQ', 'FY', 'CX', 'LM', 'TK', 'LB', 'PL', 'CC', 'SJX']
    yticks = yticks[::-1]

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
