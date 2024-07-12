import numpy as np
from config import test_case, dl_line_3_4
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import itertools


random.seed(9)


def get_row_index(array, element):
    matches = (array == np.array(element)).all(axis=1)
    if np.any(matches):
        return np.where(matches)[0][0]
    else:
        return None


# 从颜色列表随机生成颜色
def random_color():
    colors = mcolors.cnames.keys()
    return random.choice(list(colors))


"""------used in arc model-----------"""
def divide_out_int_arcs(point_index, point_arc_mapping, arc_point_mapping):
    out_arcs, in_arcs = [], []
    linked_arc_indexes = point_arc_mapping[point_index]
    for linked_arc_index in linked_arc_indexes:
        linked_point_indexes = arc_point_mapping[linked_arc_index]
        if linked_point_indexes[0] == point_index:
            out_arcs.append(linked_arc_index)
        else:
            in_arcs.append(linked_arc_index)
    return out_arcs, in_arcs


def decode_graph_to_timeslot(point_set, arc_set, selected_arc_sets_of_trains, arc_point_mapping, config):
    n_blocks = config.n_blocks
    n_trains = config.n_trains
    time_slot = np.zeros((n_blocks*2, n_trains))
    for i in range(len(selected_arc_sets_of_trains)):
        selected_arc_sets_of_train_of_i = selected_arc_sets_of_trains[i]
        for arc in selected_arc_sets_of_train_of_i:
            if arc[1] == 1:  # 运行边
                linked_block_index = arc[3]
                linked_point_indexes = arc_point_mapping[arc[0]]
                time_instance_0 = point_set[linked_point_indexes[0]][1]
                time_instance_1 = point_set[linked_point_indexes[1]][1]
                time_slot[2 * linked_block_index, i] = time_instance_0
                time_slot[2 * linked_block_index + 1, i] = time_instance_1
    return time_slot


# 检查两个列车的边集中任意一对边是否满足约束条件，如果违背了任意约束，则将该对边的编号加入不兼容集中。
def check_constraints(arc_set, block_id, arc_point_mapping, point_set, cfg):
    incompatible_arc_pairs = []
    arc_set_i, arc_set_j = arc_set[0], arc_set[1]  # 获取列车i,j的边集
    for arc_i in arc_set_i:
        index_arc_i = arc_i[0]  # 获取列车i的边的序号
        linked_points_i = arc_point_mapping[index_arc_i]  # 获取该边的连接节点编号
        # 对任意边的连接节点，用u,v分别表示位于区间后方站和前方站的节点
        u_i, v_i = point_set[linked_points_i[0]], point_set[linked_points_i[1]]
        # 获取两个节点的时刻信息
        time_u_i, time_v_i = u_i[1], v_i[1]
        for arc_j in arc_set_j:
            index_arc_j = arc_j[0]  # 获取列车j的边的序号
            linked_points_j = arc_point_mapping[index_arc_j]  # 获取该边的连接节点编号
            # 对任意边的连接节点，用u,v分别表示位于区间后方站和前方站的节点
            u_j, v_j = point_set[linked_points_j[0]], point_set[linked_points_j[1]]
            # 获取两个节点的时刻信息
            time_u_j, time_v_j = u_j[1], v_j[1]
            # 开始约束检测：
            # 1.首先检测是否违背越行约束。如果违背，将两列车的边编号加入不兼容集；否则转2.
            # 2.检测是否违背追踪间隔约束，如果违背，将两列车的边编号加入不兼容集。
            if (time_u_i - time_u_j) * (time_v_i - time_v_j) < 0:
                incompatible_arc_pairs.append([index_arc_i, index_arc_j])
            else:
                if abs(time_u_i - time_u_j) < cfg.dep_track_headway or\
                   abs(time_v_i - time_v_j) < cfg.arr_track_headway:
                    incompatible_arc_pairs.append([index_arc_i, index_arc_j])
    return incompatible_arc_pairs


# 生成点集和边集
def generate_graph(cfg):
    # 获取参数
    n_trains = cfg.n_trains
    n_stations = cfg.n_stations
    n_blocks = cfg.n_blocks
    time_zone = cfg.time_zone
    train_type = cfg.train_type
    train_speed = cfg.train_speed
    direction = cfg.direction
    distance = cfg.distance
    down_run_time = cfg.down_run_time
    up_run_time = cfg.up_run_time
    start_time = cfg.start_time
    shift_scope = cfg.shift_scope
    stop_plan = cfg.stop_plan
    punish = cfg.punish
    dep_track_headway = cfg.dep_track_headway
    arr_track_headway = cfg.arr_track_headway
    time_loss_ac = cfg.time_loss_ac
    time_loss_de = cfg.time_loss_de
    min_dwell_time = cfg.min_dwell_time
    max_dwell_time = cfg.max_dwell_time
    dwell_time_scope = [0] + list(range(min_dwell_time, max_dwell_time + 1))
    # 生成点集：1个虚拟原点 + 1个虚拟终点 + n_blocks*time_zone个出发节点 + n_blocks*time_zone个到达节点
    # 点集结构：3维矩阵  1维-对应车站序号， 2维-对应时刻， 3维-点种类
    # 点集的第1个和最后1个元素分别对应虚拟原点和虚拟终点
    # 点种类： 0-虚拟原点， 1-出发节点， 2-到达节点， 3-虚拟终点
    n_points = 2 + 2 * n_blocks * time_zone
    point_set = np.zeros((3, n_points), dtype=np.int64)
    # 生成点-边集合映射point_arc_mapping
    point_arc_mapping = [[] for _ in range(n_points)]
    # 设置虚拟原点
    point_set[0, 0] = -1
    point_set[1, 0] = -1
    point_set[2, 0] = 0
    # 设置虚拟终点
    point_set[0, -1] = -1
    point_set[1, -1] = -1
    point_set[2, -1] = 3
    # 设置出发节点和到达节点
    point_counter = 1
    for i in range(n_blocks):
        for j in range(time_zone):
            # 出发节点
            point_set[0, point_counter] = i
            point_set[1, point_counter] = j
            point_set[2, point_counter] = 1
            # 到达节点
            point_set[0, point_counter + n_blocks * time_zone] = i + 1
            point_set[1, point_counter + n_blocks * time_zone] = j
            point_set[2, point_counter + n_blocks * time_zone] = 2
            # 更新计数器
            point_counter += 1
    point_set = np.transpose(point_set)
    # 制作两个边集：1是关于每列车的边子集，以字典保存；2是所有列车的边集，由1的所有子集组合而成，以数组保存
    # 列车边集结构：4维矩阵 1维-边序号， 2维-边种类， 3维-费用, 4维-对应区间序号（运行边）或对应车站序号（停站边）
    # 边种类： 0-始发边，1-运行边，2-停站边，3-终到边
    arc_sets_of_trains = {}
    # 生成边-点集合映射
    arc_point_mapping = []
    # 边计数器
    arc_counter = 0
    for i in range(n_trains):
        arc_sets_of_trains[i] = []
        start_time_i = start_time[i]
        # 生成可行的出发时间范围
        start_time_i_lb = max(0, start_time_i + shift_scope[0])
        start_time_i_ub = min(time_zone, start_time_i + shift_scope[1] + 1)
        start_time_scope = list(range(start_time_i_lb, start_time_i_ub))
        # 生成始发边
        for j in range(len(start_time_scope)):
            # 计算始发边费用
            cost_of_start_arc = - abs(start_time_scope[j])
            arc_sets_of_trains[i].append([arc_counter,  # 边序号
                                          0,  # 边类型
                                          cost_of_start_arc,  # 边费用
                                          -1])  # 边对应车站或区间
            # 更新点-边映射和边-点映射
            point_arc_mapping[0].append(arc_counter)
            # 查询点的索引(出发点)
            point_index = get_row_index(point_set, [0,  # 始发站编号
                                                    start_time_scope[j],  # 更新后的始发时刻
                                                    1])  # 出发节点类型
            assert point_index is not None, 'the point_index is not found!'
            if point_index is not None:
                arc_point_mapping.append([0, point_index])
                point_arc_mapping[point_index].append(arc_counter)
            arc_counter += 1
            # 保存在当前车站是否停站
            stop_pre_set = [True]
            # 将当前始发点索引保存在待展开节点集中
            point_index_set_to_expand = [point_index]

            # 根据停站时间范围和当前区间待展开节点集（区间后方站的节点集）生成展开节点集（区间前方站的节点集）
            def expand_points(point_index_set_to_expand, block_index, stop_pre_set, arc_counter):
                point_index_set_expanded = []
                stop_next_set = []
                for _i in range(len(point_index_set_to_expand)):
                    point_index_to_expand = point_index_set_to_expand[_i]
                    # 获取当前时间
                    current_time = point_set[point_index_to_expand][1]
                    # 获取上一车站是否停站
                    stop_pre = stop_pre_set[_i]
                    # 生成运行边（分列车在下一车站是否停站两种情况讨论）
                    # 情况一：不停站
                    arr_time_nonstop = current_time +\
                                       stop_pre * time_loss_ac +\
                                       down_run_time[train_type[i]][block_index]
                    # 查找不停站到达点索引
                    arr_nonstop_point_index = get_row_index(point_set, [block_index + 1,  # 车站序号
                                                                        arr_time_nonstop,  # 到达时刻
                                                                        2])  # 节点类型：到达点
                    # 检查是否已经存在对应的边
                    arc = [point_index_to_expand, arr_nonstop_point_index]
                    arc_index = get_row_index(arc_point_mapping, arc)
                    if arc_index == None:
                        # 更新边集
                        # 计算不停站运行边费用
                        cost_of_run_arc = - stop_pre * time_loss_ac
                        arc_sets_of_trains[i].append([arc_counter,  # 边序号
                                                      1,  # 边类型：运行边
                                                      cost_of_run_arc,  # 费用： 起停车附加时分之和的负值
                                                      block_index])  # 前一车站序号 = 区间序号（线型铁路）
                        # 更新点-边映射和边-点映射
                        point_arc_mapping[point_index_to_expand].append(arc_counter)
                        point_arc_mapping[arr_nonstop_point_index].append(arc_counter)
                        arc_point_mapping.append(arc)
                        # 更新边计数器
                        arc_counter += 1
                    # 情况二：停站
                    arr_time_stop = current_time + \
                                    stop_pre * time_loss_ac + \
                                    down_run_time[train_type[i]][block_index] + \
                                    time_loss_de
                    # 查找到达点索引
                    arr_stop_point_index = get_row_index(point_set, [block_index + 1,  # 车站序号
                                                                     arr_time_stop,  # 到达时刻
                                                                     2])  # 节点类型：到达点
                    # 检查是否已经存在对应的边
                    arc = [point_index_to_expand, arr_stop_point_index]
                    arc_index = get_row_index(arc_point_mapping, arc)
                    if arc_index is None:
                        # 更新边集
                        # 计算停站运行边费用
                        cost_of_run_arc = - (stop_pre * time_loss_ac + time_loss_de)
                        arc_sets_of_trains[i].append([arc_counter,  # 边序号
                                                      1,  # 边类型：运行边
                                                      cost_of_run_arc,  # 费用： 起停车附加时分之和的负值
                                                      block_index])  # 前一车站序号 = 区间序号（线型铁路）
                        # 更新点-边映射和边-点映射
                        point_arc_mapping[point_index_to_expand].append(arc_counter)
                        point_arc_mapping[arr_stop_point_index].append(arc_counter)
                        arc_point_mapping.append([point_index_to_expand, arr_stop_point_index])
                        # 更新边计数器
                        arc_counter += 1
                    # 检查是否为最后一个区间（如果否，生成停站边）
                    if k != n_blocks - 1:
                        # 根据停站时间范围生成停站边（到达点-出发点）
                        # 根据停站时间范围计算出发点，然后查找出发点索引
                        for dwell_time in dwell_time_scope:
                            # 在下一车站是否停站
                            stop_next = bool(dwell_time)
                            stop_next_set.append(stop_next)
                            # 计算下一站的出发时刻（分停站和不停站两种情况讨论）
                            if stop_next is False:  # 不停站
                                dep_time = arr_time_nonstop
                                arr_point_index = arr_nonstop_point_index
                            else:  # 停站
                                dep_time = arr_time_stop + dwell_time
                                arr_point_index = arr_stop_point_index
                            # 查找出发点索引
                            dep_point_index = get_row_index(point_set, [block_index + 1,  # 车站序号
                                                                        dep_time,  # 出发时刻
                                                                        1])  # 节点类型：出发点
                            # 检查是否已经存在对应的边
                            arc = [arr_point_index, dep_point_index]
                            arc_index = get_row_index(arc_point_mapping, arc)
                            if arc_index is None:
                                # 更新边集
                                # 计算停站边费用
                                cost_of_dwell_arc = - dwell_time
                                arc_sets_of_trains[i].append([arc_counter,  # 边序号
                                                              2,  # 边类型：停站边
                                                              cost_of_dwell_arc,  # 费用： 停站时间的负值
                                                              block_index + 1])  # 当前车站序号（线型铁路）
                                # 更新点-边映射和边-点映射
                                point_arc_mapping[arr_point_index].append(arc_counter)
                                point_arc_mapping[dep_point_index].append(arc_counter)
                                arc_point_mapping.append(arc)
                                # 更新边计数器
                                arc_counter += 1
                                # 将出发节点加入已展开节点集
                                point_index_set_expanded.append(dep_point_index)
                    else:
                        # 将到达节点加入已展开节点集
                        point_index_set_expanded.append(arr_nonstop_point_index)
                        point_index_set_expanded.append(arr_stop_point_index)
                return point_index_set_expanded, stop_next_set, arc_counter

            # 根据该点的始发时刻
            # 生成各个区间(除了终到区间)的运行边和各个中间站的停站边
            for k in range(n_blocks):
                # 是否考虑速度变化？暂时不考虑
                # 生成可行的停站时间范围？ 已在设置初始始发时刻中考虑
                point_index_set_expanded, stop_next_set, arc_counter = expand_points(
                    point_index_set_to_expand, k, stop_pre_set, arc_counter)
                # 更新待展开节点集
                point_index_set_to_expand = point_index_set_expanded
                stop_pre_set = stop_next_set
            # 生成终到边（到达点-终到点）
            # 终到点索引
            end_point_index = -1
            for k in point_index_set_to_expand:
                # 检查是否已经存在对应的边
                arc = [k, end_point_index]
                arc_index = get_row_index(arc_point_mapping, arc)
                if arc_index is None:
                    # 更新边集（终到边费用为0，因此无需计算）
                    arc_sets_of_trains[i].append([arc_counter,  # 边序号
                                                  3,  # 边类型：终到边
                                                  0,  # 费用： 0
                                                  -2])
                    # 更新点-边映射和边-点映射
                    point_arc_mapping[k].append(arc_counter)
                    point_arc_mapping[end_point_index].append(arc_counter)
                    arc_point_mapping.append(arc)
                    # 更新边计数器
                    arc_counter += 1
    # 总边集结构：
    arc_sets = []
    for i in range(n_trains):
        arc_sets += arc_sets_of_trains[i]
    return point_set, arc_sets_of_trains, arc_sets, point_arc_mapping, arc_point_mapping


def draw_graph(point_set, arc_sets_of_trains, point_arc_mapping, arc_point_mapping, config):
    n_stations = config.n_stations
    n_trains = config.n_trains
    colors = [str(random_color()) for _ in range(n_trains)]
    y_gap = 10  # 纵轴间距
    # 横纵坐标
    x_ticks = np.arange(config.time_zone)
    y_ticks = np.arange(n_stations + 2) * y_gap
    y_ticks = y_ticks[::-1]
    # 画车站网格
    for i in range(n_stations):
        if i == 0 or i == n_stations - 1:
            plt.plot([0, config.time_zone-1], [y_ticks[i+1], y_ticks[i+1]], color='gray', alpha=0.5,
                     linestyle='-', linewidth=0.5)
        else:
            plt.plot([0, config.time_zone-1], [y_ticks[i+1]-1, y_ticks[i+1]-1], color='gray', alpha=0.5,
                     linestyle='-', linewidth=0.5)
            plt.plot([0, config.time_zone - 1], [y_ticks[i + 1] + 1, y_ticks[i + 1] + 1], color='gray', alpha=0.5,
                     linestyle='-', linewidth=0.5)
    # 画始发节点和终到节点
    plt.scatter(config.time_zone/2-1, y_ticks[0]-1, s=100, color='green', label='start point')
    plt.scatter(config.time_zone / 2 - 1, y_ticks[-1] + 1, s=100, color='red', label='end point')
    # 画未选择的时间节点
    for i in range(n_stations):
        for j in range(config.time_zone):
            if i == 0 or i == n_stations - 1:
                plt.scatter(j, y_ticks[i+1], s=100, color='gray')
            else:
                plt.scatter(j, y_ticks[i+1]-1, s=100, color='gray')
                plt.scatter(j, y_ticks[i+1] + 1, s=100, color='gray')

    # 获取节点在图中坐标的函数
    def get_coordinate(point):
        id_station = point[0]
        time = point[1]
        type = point[2]
        if type == 0:  # 节点为虚拟起点
            x = config.time_zone / 2 - 1
            y = y_ticks[0] - 1
        elif type == 3:  # 节点为虚拟终点
            x = config.time_zone / 2 - 1
            y = y_ticks[-1] + 1
        elif type == 1 and id_station != 0:  # 节点为出发节点
            x = time
            y = y_ticks[id_station + 1] - 1
        elif type == 2 and id_station != config.n_stations - 1:  # 节点为到达节点
            x = time
            y = y_ticks[id_station + 1] + 1
        elif type == 1 and id_station == 0:
            x = time
            y = y_ticks[1]
        elif type == 2 and id_station == config.n_stations - 1:
            x = time
            y = y_ticks[-2]
        return x, y

    # 画已选择的节点和边（以列车为单位）
    for i in range(len(arc_sets_of_trains)):
        arc_set_of_train_i = arc_sets_of_trains[i]
        # 获取当前列车涉及的边序号范围
        arc_index_scope = [arc[0] for arc in arc_set_of_train_i]
        selected_arc_point_mapping = np.array(arc_point_mapping)[arc_index_scope]
        # 根据边-点映射确定当前列车涉及的点序号范围
        unique_selected_points = np.unique(list(itertools.chain.from_iterable(
            selected_arc_point_mapping)))
        # 去除虚拟起点（0）和虚拟终点(-1)
        unique_selected_points = unique_selected_points[unique_selected_points != 0]
        unique_selected_points = unique_selected_points[unique_selected_points != -1]
        # 画边
        for j in range(len(selected_arc_point_mapping)):
            linked_points = selected_arc_point_mapping[j]
            # 获取边的两端节点：s_-起点，e_-终点
            s_ = point_set[linked_points[0]]
            e_ = point_set[linked_points[1]]
            # 获取s_的x,y坐标
            s_x, s_y = get_coordinate(s_)
            e_x, e_y = get_coordinate(e_)
            plt.plot([s_x, e_x], [s_y, e_y], color=colors[i])
        # 画点
        for point_index in unique_selected_points:
            point_x, point_y = get_coordinate(point_set[point_index])
            plt.scatter(point_x, point_y, s=100, color=colors[i])
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.legend()
    plt.show()


# 根据参数和边集生成不兼容集
def generate_incompatible_sets(cfg, point_set, arc_sets_of_trains, point_arc_mapping, arc_point_mapping):
    incompatible_sets = []
    n_trains = cfg.n_trains
    n_blocks = cfg.n_blocks
    # 比较每一对列车在各个区间中的运行边，如果任意一对边违背了约束则将它们加入不兼容集中
    for i in range(n_trains - 1):
        arc_set_of_train_i = arc_sets_of_trains[i]  # 获取第i个列车的边集
        run_arc_i = [arc for arc in arc_set_of_train_i if arc[1] == 1]  # 获取第i个列车的运行边集
        # 获取当前列车运行边的边序号
        run_arc_index_i = [arc[0] for arc in run_arc_i]
        for j in range(i+1, n_trains):
            arc_sets_of_train_j = arc_sets_of_trains[j]  # 获取第j个列车的边集
            run_arc_j = [arc for arc in arc_sets_of_train_j if arc[1] == 1]  # 获取第j个列车的运行边集
            for k in range(n_blocks):
                # 获取两个运行边集中对应区间k的运行边
                run_arc_i_k = [arc for arc in run_arc_i if arc[-1] == k]
                run_arc_j_k = [arc for arc in run_arc_j if arc[-1] == k]
                # test
                # draw_graph(point_set, {0: run_arc_i_k, 1: run_arc_j_k},
                #            point_arc_mapping, arc_point_mapping, config)
                incompatible_sets += check_constraints((run_arc_i_k, run_arc_j_k),
                                                       k,
                                                       arc_point_mapping,
                                                       point_set,
                                                       cfg)
    return incompatible_sets


# def decode_solution_to_timeslot(solution, arc_point_mapping, point_set, arc_sets_of_trains, arc_set, config):
#     n_trains = config.n_trains
#     n_blocks = config.n_blocks
#     time_slot = np.zeros((2*n_blocks, n_trains))
#     # 从solution中获取值为1的边索引: 运行边
#     selected_arcs_indexes = np.where(solution == 1)[0]
#     for selected_arcs_index in selected_arcs_indexes:
#         # 获取边相关的列车和区间编号
#         for key, values in arc_sets_of_trains.items():
#             if selected_arcs_index in values:
#                 train_id = key
#                 arc_set_of_train = arc_sets_of_trains[key]
#         block_id = arc_set[selected_arcs_index][-1]
#         # 通过运行边获取连接节点的索引
#         linked_points_index = arc_point_mapping[selected_arcs_index]
#         point_1 = point_set[linked_points_index[0]]
#         point_2 = point_set[linked_points_index[1]]
#         # 获取两个连接节点的时刻
#         time_1, time_2 = point_1[1], point_2[1]
#         # 根据区间编号和列车编号更新
#         time_slot[2*block_id, train_id] = point_1
#         time_slot[2*block_id, train_id] = point_2
#     return time_slot


"""------used in node model-----------"""


# 根据列车边集和节点集生成列车节点集
def generate_point_sets_of_trains(point_set, arc_sets_of_trains, arc_point_mapping):
    n_trains = len(arc_sets_of_trains)
    point_sets_of_trains = {}
    point_indices_of_trains = {}
    for i in range(n_trains):
        arc_sets_of_train_i = arc_sets_of_trains[i]  # 列车i的边集
        # 根据列车i的边集生成列车i的边编号集
        arc_indices_of_train_i = [arc_of_train_i[0] for arc_of_train_i in arc_sets_of_train_i]
        # 根据列车i的边编号和边-节点映射生成列车i的节点编号集
        point_indices_duplicate = (np.array(arc_point_mapping)[arc_indices_of_train_i]).flatten()
        point_indices = np.unique(point_indices_duplicate)
        # 去除虚拟始发节点和终到节点编号
        point_indices = point_indices[point_indices != 0]
        point_indices = point_indices[point_indices != -1]
        # 根据节点集和列车i的节点编号集生成列车i的节点集
        point_indices_of_trains[i] = point_indices
        point_sets_of_trains[i] = point_set[point_indices]
    return point_sets_of_trains, point_indices_of_trains


# 根据参数和点集生成不兼容集
def generate_incompatible_sets_of_points(cfg, point_set, arc_sets_of_trains, point_arc_mapping, arc_point_mapping):
    incompatible_sets = []
    return incompatible_sets


# test code
if __name__ == '__main__':
    config = test_case()
    # config = dl_line_3_4()
    point_set, arc_sets_of_trains, arc_sets, point_arc_mapping, arc_point_mapping = \
        generate_graph(config)
    point_sets_of_trains, point_indices_of_trains =\
        generate_point_sets_of_trains(point_set, arc_sets_of_trains, arc_point_mapping)
    # draw_graph(point_set, arc_sets_of_trains, point_arc_mapping, arc_point_mapping, config)
    # incompatible_sets = generate_incompatible_sets(config, point_set, arc_sets_of_trains, point_arc_mapping, arc_point_mapping)