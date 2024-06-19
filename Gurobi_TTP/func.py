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


# def e2m(path):
#     path = '..\\General_file\\Data\\' + path
#     table = xlrd.open_workbook(path).sheets()[0]# get the first sheet from the Excel file
#     row = table.nrows
#     col = table.ncols
#     datamatrix = np.zeros((row, col))# generate a matrix (@row * @col) filled with 0
#     for x in range(col):
#         cols = np.matrix(table.col_values(x))
#         datamatrix[:, x] = cols
#     return datamatrix

# test data
# stop_time = np.array([[5, 0, 5],
#                       [0, 5, 5]])
# origin_time = np.array([0, 15, 30])
# turn_head = {0: (5,), 1: (5,), 2: (5,)}
# running_time = np.array([10, 10, 10])
# tl_ac, tl_dc = 2, 3
# params = {'running_time': running_time, 'tl_ac': tl_ac, 'tl_dc': tl_dc}


def get_ts_from_info(stop_time, origin_time, turn_head, params):

    # get time_slot from information:
    # Input: stop_plan, origin_time, turn_head, params(a dictionary,
    # including-{running_time, tl_ac, tl_dc})
    # stop_time: 2-D array, size=(num_im_s, num_rs)
    # origin_time: 1-D array, size=(num_rs,)
    # turn_head: dictionary, size=d{turn_times(0), turn_times(1),..., turn_times(num_rs)}
    # running_time: 1-D array, size=(num_b,)
    # tl_ac, tl_dc: scalar
    # Output: time_slot, rolling_stock
    # time_slot: 2-D array, size=(num_t, (num_b+1)*2)
    # rolling_stock: dictionary, size=d{turn_times(0)+1, turn_times(1)+1,..., turn_times(num_rs)+1}

    # region 输入数据安全检查
    assert type(stop_time) is np.ndarray
    assert type(origin_time) is np.ndarray
    assert isinstance(turn_head, dict)

    assert isinstance(params, dict)
    assert 'running_time' in params.keys()
    assert 'tl_ac' in params.keys()
    assert 'tl_dc' in params.keys()
    assert type(params['running_time']) is np.ndarray

    assert np.ndim(stop_time) == 2
    assert np.ndim(stop_time) == 2
    assert np.ndim(origin_time) == 1
    assert np.ndim(params['running_time']) == 1
    assert np.ndim(params['tl_ac']) == 0
    assert np.ndim(params['tl_dc']) == 0
    assert len(params['running_time']) == np.shape(stop_time)[0] + 1
    # endregion

    num_im_s, num_rs = np.shape(stop_time)
    num_s = num_im_s + 2
    # print(num_im_s)
    num_b = num_s - 1
    # print(num_im_s, num_rs)

    turn_times = np.array([len(turn_head[i]) for i in range(num_rs)])
    print(turn_times)
    num_t = np.sum(turn_times + 1)
    print(num_t)

    running_time = params['running_time']
    tl_ac = params['tl_ac']
    tl_dc = params['tl_dc']

    down_run_time = running_time
    up_run_time = np.flip(down_run_time)
    down_tl_ac = tl_ac
    down_tl_dc = tl_dc
    up_tl_ac = tl_dc
    up_tl_dc = tl_ac

    down_stop_time = stop_time
    up_stop_time = stop_time[::-1, :]

    down_whether_stop = stop_time > 0
    _add = np.array([[True for i in range(num_rs)]])
    down_whether_stop = np.vstack((_add, down_whether_stop, _add))
    up_whether_stop = down_whether_stop[::-1, :]

    time_slot = np.zeros((num_b * 2, num_t))
    rolling_stock = {}

    # generate rolling_stock
    id_t = 0
    id_rs = 0
    while id_t < num_t:
        duplicated_turn_times = turn_times
        if id_t < num_rs:
            rolling_stock[id_t] = [id_t]
            id_t += 1
            id_rs = (id_rs + 1) % num_rs
        else:
            if duplicated_turn_times[id_rs] > 0:
                rolling_stock[id_rs].append(id_t)
                duplicated_turn_times[id_rs] -= 1
                id_t += 1
                id_rs = (id_rs + 1) % num_rs
            else:
                id_rs = (id_rs + 1) % num_rs

    # generate the direction of all trains
    direction = np.zeros(num_t)
    for i in range(num_rs):
        for j in range(len(rolling_stock[i])):
            if j % 2 == 0:
                direction[rolling_stock[i][j]] = 0
            else:
                direction[rolling_stock[i][j]] = 1

    # generate time_slot
    id_t = 0
    id_rs = 0
    while id_t < num_t:

        if id_t < num_rs:
            init_time = origin_time[id_t]
            id_rs = id_t
            id_turn = 0
        else:
            # get the init_time of present train by adding the arrival time of the previous train using the
            # same rolling stock at the destination station and corresponding turn headway.
            for i in range(num_rs):
                if id_t in rolling_stock[i]:
                    id_pre_t = rolling_stock[i][rolling_stock[i].index(id_t) - 1]
                    # id_pre_t = rolling_stock[i].index(id_t) - 1
                    print(id_pre_t, i)
                    turn_time = turn_head[i][rolling_stock[i].index(id_t) - 1]
                    id_rs = i
                    id_turn = rolling_stock[i].index(id_t) - 1
            init_time = time_slot[-1, id_pre_t] + turn_time

        if direction[id_t] == 0:  # downstream train
            time_slot[0, id_t] = init_time
        else:  # upstream train
            time_slot[-1, id_t] = init_time

        for i in range(num_b - 1):
            if direction[id_t] == 0:  # downstream train
                id_arr_time = 2 * i + 1
                id_dep_time = 2 * i + 2
                arr_time = time_slot[id_arr_time - 1, id_t] + down_run_time[i] + down_whether_stop[i][
                    id_rs] * down_tl_ac + \
                           down_whether_stop[i + 1][id_rs] * down_tl_dc
                # print(arr_time)
                dep_time = arr_time + down_stop_time[i][id_rs]
                # print(dep_time)
                time_slot[id_arr_time, id_t] = arr_time
                time_slot[id_dep_time, id_t] = dep_time
            else:  # upstream train
                id_arr_time = (num_b * 2 - 1) - (2 * i + 1)
                id_dep_time = (num_b * 2 - 1) - (2 * i + 2)
                arr_time = time_slot[id_arr_time + 1, id_t] + up_run_time[i] + up_whether_stop[i][id_rs] * up_tl_ac + \
                           up_whether_stop[i + 1][id_rs] * up_tl_dc
                dep_time = arr_time + up_stop_time[i][id_rs]
                time_slot[id_arr_time, id_t] = arr_time
                time_slot[id_dep_time, id_t] = dep_time

        if direction[id_t] == 0:  # downstream train
            time_slot[-1, id_t] = time_slot[-2, id_t] + down_run_time[num_b - 1] + down_whether_stop[num_b - 1][id_rs]\
                                  * down_tl_ac + down_tl_dc
        else:  # upstream train
            time_slot[0, id_t] = time_slot[1, id_t] + up_run_time[num_b - 1] + up_whether_stop[num_b - 1][id_rs]\
                                 * up_tl_ac + up_tl_dc

        id_t += 1

    return time_slot, rolling_stock, direction


def unpack_time_slot(dep_time, arr_time, num_t, num_b):
    dep_time_slot = np.reshape(dep_time, (num_b, num_t), order='f')
    arr_time_slot = np.reshape(arr_time, (num_b, num_t), order='f')
    time_slot = []
    for i in range(num_b):
        time_slot.append(dep_time_slot[i])
        time_slot.append(arr_time_slot[i])
    time_slot = np.array(time_slot)
    return time_slot, dep_time_slot, arr_time_slot



# test code


# time_slot, rolling_stock, direction = get_ts_from_info(stop_time, origin_time, turn_head, params)
# print('time_slot: \n' + str(time_slot))
# print('rolling_stock: \n' + str(rolling_stock))







