import numpy as np
import itertools
import config
from config import test_case
from general_func import draw, write_file
import gurobipy as gp
from gurobipy import GRB, LinExpr
from func import generate_graph, draw_graph, generate_incompatible_sets, divide_out_int_arcs,\
    decode_graph_to_timeslot, generate_point_sets_of_trains


def main(cfg):
    # 生成图和不兼容集
    point_set, arc_sets_of_trains, arc_set, point_arc_mapping, arc_point_mapping = \
        generate_graph(cfg)
    point_sets_of_trains, point_indices_of_trains =\
        generate_point_sets_of_trains(point_set, arc_sets_of_trains, arc_point_mapping)
    # 获取参数
    n_arcs = len(arc_set)
    # 获取所有已选择节点的索引，并统计总数
    list_ = [point_indices_of_train.tolist() for point_indices_of_train in point_indices_of_trains.values()]
    indices_selected_points = np.unique(list(itertools.chain.from_iterable(list_)))
    n_points = len(indices_selected_points)
    n_trains = cfg.n_trains
    # 建立模型
    model = gp.Model()
    # 定义决策变量
    x_a = {}
    for i in range(n_arcs):
        name = 'x_' + str(i)
        x_a[i] = model.addVar(0, 1, vtype=GRB.BINARY, name=name)
    y_v = {}
    for i in range(n_points):
        name = 'y_' + str(i)
        y_v[i] = model.addVar(0, GRB.INFINITY, vtype=GRB.INTEGER, name=name)  # 论文中是0-1变量，但理论上应该是整数变量
    z_j_v = {}
    for i in range(n_trains):
        for j in range(n_points):
            name = 'z_' + str(i) + '_' + str(j)
            z_j_v[i, j] = model.addVar(0, 1, vtype=GRB.BINARY, name=name)
    # 更新模型
    model.update()
    # 定义目标函数
    obj = 0
    for i in range(n_arcs):
        obj += x_a[i] * arc_set[i][2]
    model.setObjective(obj, GRB.MAXIMIZE)
    # 定义约束条件
    # 1. 唯一始发边约束
    for i in range(n_trains):
        start_arc_indexes = [arc[0] for arc in arc_sets_of_trains[i] if arc[1] == 0]
        constr = 0
        for start_arc_index in start_arc_indexes:
            constr += x_a[start_arc_index]
        model.addConstr(constr - 1 == 0)
    # 2. 流平衡约束
    for i in range(n_trains):
        # 获取每列车的出发节点和到达节点索引:
        # 获取当前列车涉及的边序号范围
        arc_index_scope = [arc[0] for arc in arc_sets_of_trains[i]]
        selected_arc_point_mapping = np.array(arc_point_mapping)[arc_index_scope]
        # 根据边-点映射确定当前列车涉及的点序号范围
        unique_selected_points = np.unique(list(itertools.chain.from_iterable(
            selected_arc_point_mapping)))
        # 去除虚拟起点（0）和虚拟终点(-1)
        unique_selected_points = unique_selected_points[unique_selected_points != 0]
        unique_selected_points = unique_selected_points[unique_selected_points != -1]
        # 对当前列车的每一个出发节点和到达节点，分别获取其所有入边和出边的编号，两类边的和需要相等
        for point_index in unique_selected_points:
            out_arcs, in_arcs = divide_out_int_arcs(point_index, point_arc_mapping, arc_point_mapping)
            out_sums, in_sums = 0, 0
            for out_arc in out_arcs:
                out_sums += x_a[out_arc]
            for in_arc in in_arcs:
                in_sums += x_a[in_arc]
            model.addConstr(out_sums == in_sums)
    # 3. 不兼容集约束
    # 生成不兼容节点集

    return
