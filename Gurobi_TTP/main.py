from gurobipy import *
import numpy as np

from func import write_file
from general_func import draw


def main():
    # 区间纯运行时分
    tr_down_b = [9, 8, 7]
    tr_up_b = [9, 8, 7]
    # 站间距
    distance = [8.1, 10.2, 9]
    # distance = runningTime
    # 列车i的最早始发时刻
    te_i = [0, 17, 47]
    # 列车i的最晚始发时刻
    tl_i = te_i
    # 罚值
    M = 10e5

    num_I = 3  # 列车数
    num_U = 4  # 车站数
    num_B = num_U - 1  # 区间数
    # num_G = 2   # 列车种类总数
    # num_U_i = 0  # 列车i途经的车站集
    # num_U_i_hat = 0  # 列车i必须停站的车站集
    num_N = 60   # 规划时间段数（min）
    # num_N = 1440 - 6 * 60   # 规划时间段数（min）
    z_i = [0, 0, 1]  # 0-下行列车；1-上行列车
    # z_i = [0, 0, 0, 1, 1, 1]
    # z_i = [0 for _ in range(13)] + [1 for _ in range(13)]

    h_tli = 4   # 连发间隔
    h_tb = 2   # 不同时到达间隔
    h_th = 2   # 会车间隔
    t_ac = 2   # 启动附加时分
    t_dc = 3   # 停车附加时分
    tw_min = 4  # 列车i在车站u的最短停站时间
    tw_max = 25  # 列车i在车站u的最长停站时间

    model = Model()
    # 定义决策变量
    td_i_b = {}  # 列车i在区间b后方站的出发时刻
    ta_i_b = {}  # 列车i在区间b前方站的到达时刻
    y_i_b = {}  # 列车i在车站的停站情况，停站：-1；通过：-0,始发终到站默认为停站
    s_i_j_b = {}  # 列车i和列车j在区间b中的次序变量，若列车i在列车j之前则为1，反之则为0
    u_i_j_b = {}  # 列车i和列车j在中间站b的次序变量，若列车i在列车j之前则为1，反之则为0
    g_a_b_c = {}
    o_a_b_c = {}
    p = {}  # 列车i在时刻t是否占据区间b，若是则为1，否则为0
    for i in range(num_I):
        for b in range(num_B):
            name = 'td_'+str(i)+'_'+str(b)
            td_i_b[i, b] = model.addVar(0, num_N*60, vtype=GRB.INTEGER, name=name)
    for i in range(num_I):
        for b in range(num_B):
            name = 'ta_'+str(i)+'_'+str(b)
            ta_i_b[i, b] = model.addVar(0, num_N*60, vtype=GRB.INTEGER, name=name)
    for i in range(num_I):
        for b in range(num_B-1):
            name = 'y_'+str(i)+'_'+str(b)
            y_i_b[i, b] = model.addVar(0, 1, vtype=GRB.BINARY, name=name)
    for i in range(num_I):
        for j in range(num_I):
            if i == j:
                continue
            for b in range(num_B):
                name = 's_' + str(i) + '_' + str(j) + '_' + str(b)
                s_i_j_b[i,j,b] = model.addVar(0,1,vtype=GRB.BINARY,name=name)
    for i in range(num_I):
        for j in range(num_I):
            if i == j:
                continue
            for b in range(num_B-1):
                name = 's_' + str(i) + '_' + str(j) + '_' + str(b)
                u_i_j_b[i,j,b] = model.addVar(0,1,vtype=GRB.BINARY,name=name)

    for i in range(num_I):
        for j in range(num_B):
            for k in range(num_N):
                name = 'g_' + str(i) + '_' + str(j) + '_' + str(k)
                g_a_b_c[i,j,k] = model.addVar(0,1,vtype=GRB.BINARY)

    for i in range(num_I):
        for j in range(num_B):
            for k in range(num_N):
                name = 'o_' + str(i) + '_' + str(j) + '_' + str(k)
                o_a_b_c[i,j,k] = model.addVar(0,1,vtype=GRB.BINARY)

    for i in range(num_I):
        for j in range(num_B):
            for k in range(num_N):
                name = 'p_' + str(i) + '_' + str(j) + '_' + str(k)
                p[i,j,k] = model.addVar(0,1,vtype=GRB.BINARY)

    model.update()

    # 定义目标函数
    obj = LinExpr(0)
    for i in range(num_I):
        if z_i[i] == 0: # 下行列车
            obj.addTerms(1, ta_i_b[i,num_B-1])
            obj.addTerms(-1, td_i_b[i,0])
        else: # 上行列车
            obj.addTerms(-1, ta_i_b[i, num_B - 1])
            obj.addTerms(1, td_i_b[i, 0])
    model.setObjective(obj, GRB.MINIMIZE)
    print(model.getObjective())

    ### 定义约束条件
    # 运行时间约束
    for i in range(num_I):
        for b in range(num_B):
            if z_i[i] == 0: # 下行列车
                if b == 0: # 初始区间
                    model.addConstr(ta_i_b[i,b] - td_i_b[i,b] == tr_down_b[b] + t_ac + y_i_b[i, b] * t_dc, name='downstream train_'+str(i)+' running time constr at station_'+str(b))
                elif b == num_B - 1:
                    model.addConstr(ta_i_b[i, b] - td_i_b[i, b] == tr_down_b[b] + y_i_b[i, b - 1] * t_ac + t_dc,
                                    name='downstream train_' + str(i) + ' running time constr at station_' + str(b))
                else:
                    model.addConstr(ta_i_b[i, b] - td_i_b[i, b] == tr_down_b[b] + y_i_b[i, b - 1] * t_ac + y_i_b[i, b] * t_dc,
                                    name='downstream train_' + str(i) + ' running time constr at station_' + str(b))
            else: # 上行列车
                if b == 0:
                    model.addConstr(td_i_b[i,b] - ta_i_b[i,b] == tr_up_b[b] + t_dc + y_i_b[i, b] * t_ac, name='upstream train_'+str(i)+' running time constr at station_'+str(b))
                elif b == num_B - 1:
                    model.addConstr(td_i_b[i, b] - ta_i_b[i, b] == tr_up_b[b] + y_i_b[i, b - 1] * t_dc + t_ac,
                                    name='upstream train_' + str(i) + ' running time constr at station_' + str(b))
                else:
                    model.addConstr(td_i_b[i, b] - ta_i_b[i, b] == tr_up_b[b] + y_i_b[i, b - 1] * t_dc + y_i_b[i, b] * t_ac, name='upstream train_' + str(i) + ' running time constr at station_' + str(b))


    # 始发时刻约束
    for i in range(num_I):
        model.addConstr(td_i_b[i, 0] >= te_i[i], name='lower bound of start time')
        model.addConstr(td_i_b[i, 0] <= tl_i[i], name='upper bound of start time')


    # 停站时间约束
    for i in range(num_I):
        for b in range(num_B-1):
            if z_i[i] == 0: # 下行列车
                model.addConstr(td_i_b[i,b+1] - ta_i_b[i,b] >= y_i_b[i, b] * tw_min, name='lower bound of dwell time for downstream trains')
                model.addConstr(td_i_b[i,b+1] - ta_i_b[i,b] <= y_i_b[i, b] * tw_max, name='upper bound of dwell time for downstream trains')
            else:  # 上行列车
                model.addConstr(ta_i_b[i,b] - td_i_b[i,b+1] >= y_i_b[i, b] * tw_min, name='lower bound of dwell time for upstream trains')
                model.addConstr(ta_i_b[i,b] - td_i_b[i,b+1] <= y_i_b[i, b] * tw_max, name='upper bound of dwell time for upstream trains')


    #  不同时到达时间间隔约束
    for i in range(num_I):
        for j in range(num_I):
            if i == j or z_i[i] == z_i[j]:
                continue
            for b in range(num_B-1):
                model.addConstr(ta_i_b[i, b] - td_i_b[j, b+1] + M * u_i_j_b[i, j, b] >= h_tb, name='tao bu 1')
                model.addConstr(td_i_b[j, b+1] - ta_i_b[i, b] - M * u_i_j_b[i, j, b] >= h_tb-M, name='tao bu 2')


    # 辅助变量u_i_j_b相关约束
    # for i in range(num_I):
    #     for j in range(num_I):
    #         if i == j:
    #             continue
    #         for b in range(num_B - 1):
    #             model.addGenConstrIndicator(g_i_j_b[i, j, b], 1, ta_i_b[i, b] - td_i_b[j, b + 1] <= 0)
    #             model.addGenConstrIndicator(g_i_j_b[i, j, b], 0, ta_i_b[i, b] - td_i_b[j, b + 1] >= 0)


    # 会车时间间隔约束
    for i in range(num_I):
        for j in range(num_I):
            if i == j or z_i[i] == z_i[j]:
                continue
            for b in range(num_B-1):
                model.addConstr(ta_i_b[j, b] - ta_i_b[i, b] - M * s_i_j_b[i, j, b] >= h_th - M, name='tao hui 1')
                model.addConstr(td_i_b[i, b + 1] - td_i_b[j, b + 1] + M * s_i_j_b[i, j, b + 1] >= h_th, name='tao hui 2')

    # 连发时间间隔约束
    for i in range(num_I):
        for j in range(num_I):
            if i == j or z_i[i] != z_i[j]:
                continue
            for b in range(num_B):
                if z_i[i] == 0 and z_i[j] == 0:
                    model.addConstr(td_i_b[j, b] - ta_i_b[i, b] + (1 - s_i_j_b[i, j, b]) * M >= h_tli)
                elif z_i[i] == 1 and z_i[j] == 1:
                    model.addConstr(ta_i_b[j, b] - td_i_b[i, b] + (1 - s_i_j_b[i, j, b]) * M >= h_tli)

    # 区间不交叉约束
    # 辅助变量p,g,o的相关约束
    for i in range(num_I):
        for j in range(num_B):
            for k in range(num_N):
                if z_i[i] == 0:
                    model.addConstr(g_a_b_c[i,j,k]*M >= ta_i_b[i,j]-(k-1))
                    model.addConstr((1-g_a_b_c[i,j,k])*M >= (k-1)-ta_i_b[i,j])

                    model.addConstr(o_a_b_c[i, j, k] * M >= (k+1) - td_i_b[i, j])
                    model.addConstr((1 - o_a_b_c[i, j, k]) * M >= td_i_b[i, j] - (k+1))
                else:
                    model.addConstr(g_a_b_c[i, j, k] * M >= td_i_b[i, j] - (k - 1))
                    model.addConstr((1 - g_a_b_c[i, j, k]) * M >= (k - 1) - td_i_b[i, j])

                    model.addConstr(o_a_b_c[i, j, k] * M >= (k + 1) - ta_i_b[i, j])
                    model.addConstr((1 - o_a_b_c[i, j, k]) * M >= ta_i_b[i, j] - (k + 1))

                model.addGenConstrAnd(p[i, j, k], [g_a_b_c[i, j, k], o_a_b_c[i, j, k]])

    # 不交叉约束
    lhs = LinExpr(0)

    for b in range(num_B):
        for t in range(num_N):
            for i in range(num_I):
                lhs.addTerms(1, p[i,b,t])
            model.addConstr(lhs <= 1)
            lhs.clear()

    model.setParam('OutputFlag', 1)
    solution = model.optimize()
    if model.Status == 2:
        td_opt = np.zeros((num_I, num_B))
        ta_opt = np.zeros((num_I, num_B))
        y_opt = np.zeros((num_I, num_B-1))
        s_opt = np.zeros((num_I, num_I, num_B))
        # g_opt = np.zeros((num_I, num_I, num_N*60))
        # o_opt = np.zeros((num_I, num_B, num_N*60))
        # p_opt = np.zeros((num_I, num_B, num_N*60))
        for item in model.getVars():
            if item.varname[:2] == 'td':
                id_i = int(item.varname[3])
                id_b = int(item.varname[5])
                td_opt[id_i, id_b] = item.x
            elif item.varname[:2] == 'ta':
                id_i = int(item.varname[3])
                id_b = int(item.varname[5])
                ta_opt[id_i, id_b] = item.x
            elif item.varname[:2] == 'y_':
                id_i = int(item.varname[2])
                id_b = int(item.varname[4])
                y_opt[id_i, id_b] = item.x
            elif item.varname[:2] == 's_':
                id_i = int(item.varname[2])
                id_j = int(item.varname[4])
                id_b = int(item.varname[6])
                s_opt[id_i, id_j, id_b] = item.x
            # elif item.varname[:2] == 'g_':
            #     id_i = int(item.varname[2])
            #     id_j = int(item.varname[4])
            #     id_b = int(item.varname[6])
            #     g_opt[id_i, id_j, id_b] = item.x
            # elif item.varname[:2] == 'o_':
            #     id_i = int(item.varname[2])
            #     id_b = int(item.varname[4])
            #     id_t = int(item.varname[6])
            #     o_opt[id_i, id_b, id_t] = item.x
            # elif item.varname[:2] == 'p_':
            #     id_i = int(item.varname[2])
            #     id_b = int(item.varname[4])
            #     id_t = int(item.varname[6])
            #     p_opt[id_i, id_b, id_t] = item.x



        # print('td_opt:\n'+str(td_opt))
        # print('\nta_opt:\n'+str(ta_opt))
        # print('\ny_opt:\n'+str(y_opt))
        # print('\ns_opt:\n'+str(s_opt))
        # print('\ng_opt:\n' + str(g_opt))
        # print('\no_opt:\n' + str(o_opt))
        # print('\np_opt:\n' + str(p_opt))


        time_slot = []
        for i in range(num_B):
            row_td = [td_opt[j][i] for j in range(num_I)]
            row_ta = [ta_opt[j][i] for j in range(num_I)]
            time_slot.append(row_td)
            time_slot.append(row_ta)
        time_slot = np.array(time_slot)
        print('time_slot:\n'+str(time_slot))
        write_file('time_slot_3_4_gurobi', time_slot)
        draw(distance=distance, num_S=num_U, time_slot=time_slot, time_horizon=num_N)


if __name__ == "__main__":
    main()
