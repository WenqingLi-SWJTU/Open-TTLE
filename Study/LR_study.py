import numpy as np
from scipy.optimize import minimize


def relaxed_objective(x, lambda_):
    # 放松后的目标函数
    x1, x2 = x
    return - x1 - 2 * x2 + lambda_ * (x1 + x2 - 2)  # 注意：我们取负号，因为scipy.optimize.minimize默认求解最小值问题


def solve_relaxed_problem(lambda_):
    # 求解放松后的问题
    constraints = ({'type': 'ineq', 'fun': lambda x: x[0]},  # x1 >= 0
                   {'type': 'ineq', 'fun': lambda x: x[1]})  # x2 >= 0
    result = minimize(relaxed_objective, [0, 0], args=(lambda_), method='SLSQP', constraints=constraints)
    return result.x, -result.fun  # 返回解和对应的目标函数值（取负号还原为正数）


def lagrange_relaxation():
    # 拉格朗日松弛算法
    lambda_ = 0.0  # 初始化拉格朗日乘子
    epsilon = 1e-6  # 收敛精度
    max_iter = 1000  # 最大迭代次数

    for iter_ in range(max_iter):
        x, obj_val = solve_relaxed_problem(lambda_)
        if x[0] + x[1] > 2:  # 如果解违反了原约束1
            lambda_ += epsilon  # 增加拉格朗日乘子以惩罚违反约束的解
        else:
            break  # 如果解满足原约束1，则停止迭代

    return x, obj_val


# 运行拉格朗日松弛算法
x, obj_val = lagrange_relaxation()
print(f"Solution: x1 = {x[0]}, x2 = {x[1]}, Objective Value = {obj_val}")