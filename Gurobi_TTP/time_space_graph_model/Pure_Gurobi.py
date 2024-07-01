import numpy as np
import itertools
import config
from config import test_case
from general_func import draw, write_file
import gurobipy as gp
from gurobipy import GRB, LinExpr
from func import generate_graph, draw_graph, generate_incompatible_sets, divide_out_int_arcs,\
    decode_graph_to_timeslot


def main(cfg):
    # 生成图和不兼容集
    point_set, arc_sets_of_trains, arc_set, point_arc_mapping, arc_point_mapping = \
        generate_graph(cfg)
    # 获取参数
    n_points = len(point_set)
    n_arcs = len(arc_set)
    n_trains = cfg.n_trains
    # 建立模型
    model = gp.Model()
    # 定义决策变量
    x_a = {}
    for i in range(n_arcs):
        name = 'x_' + str(i)
        x_a[i] = model.addVar(0, 1, vtype=GRB.BINARY, name=name)
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
    # 生成不兼容边集
    incompatible_arcs = generate_incompatible_sets(cfg, point_set, arc_sets_of_trains, point_arc_mapping,
                                                   arc_point_mapping)
    # 为不兼容边集中的每一个子集生成一个约束条件
    for arc_sets in incompatible_arcs:
        sums = 0
        for arc_index in arc_sets:
            sums += x_a[arc_index]
        model.addConstr(sums <= 1)

    model.setParam('OutputFlag', 1)
    model.optimize()
    # 遍历并打印所有约束
    # for constr in model.getConstrs():
    #     print(model.getRow(constr))

    if model.Status == 2:
        x_opt = np.zeros(n_arcs,)
        for item in model.getVars():
            id = int(item.varname[2:])
            x_opt[id] = item.x
        selected_arcs_index = np.where(x_opt == 1)[0]
        selected_arc_sets_of_trains = {}
        arc_set = np.array(arc_set)
        for i in range(len(arc_sets_of_trains)):
            arc_index_scope_i = [arc[0] for arc in arc_sets_of_trains[i]]
            selected_arcs_index_i = [x for x in selected_arcs_index if x in arc_index_scope_i]
            selected_arc_sets_of_trains[i] = arc_set[selected_arcs_index_i]
        # print(selected_arcs_index)
        time_slot = decode_graph_to_timeslot(point_set, arc_set, selected_arc_sets_of_trains,
                                             arc_point_mapping, cfg)
        # draw_graph(point_set, selected_arc_sets_of_trains, point_arc_mapping, arc_point_mapping, cfg)
        draw(time_slot, cfg)
    return time_slot



if __name__ == '__main__':
    # cfg = config.test_case()
    cfg = config.dl_line_3_4()
    time_slot = main(cfg)
    print(time_slot)
