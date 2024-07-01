import numpy as np
import config
from func import generate_graph, draw_graph, generate_incompatible_sets

# 一.获取案例参数
# 二.生成点集和边集
# 1. 点集：3维矩阵  1维-序号， 2维-连接边序号， 3维-点种类
# 2. 边集：3维矩阵  1维-序号， 2维-连接点序号， 3维-边种类
# 3. 精简点集和边集（可能不需要精简弧段集？）
# 三.根据边种类和案例参数计算边费用
# 四.根据约束条件生成不兼容集
# 五.建立模型并求解


class Lagrangian_Relaxation:
    def __init__(self, cfg):
        self.exp_name = cfg.exp_name
        self.n_trains = cfg.n_trains
        self.n_stations = cfg.n_stations
        self.n_blocks = cfg.n_blocks
        self.time_zone = cfg.time_zone
        self.train_type = cfg.train_type
        self.train_speed = cfg.train_speed
        self.direction = cfg.direction
        self.distance = cfg.distance
        self.down_run_time = cfg.down_run_time
        self.up_run_time = cfg.up_run_time
        self.start_time = cfg.start_time
        self.shift = cfg.shift
        self.stop_plan = cfg.stop_plan
        self.punish = cfg.punish
        self.dep_track_headway = cfg.dep_track_headway
        self.arr_track_headway = cfg.arr_track_headway
        self.time_loss_ac = cfg.time_loss_ac
        self.time_loss_de = cfg.time_loss_de
        self.min_dwell_time = cfg.min_dwell_time
        self.max_dwell_time = cfg.max_dwell_time

        # 生成点集和边集
        self.point_set, self.arc_set = generate_point_arc_sets(cfg)
        # 根据边种类计算所有边的费用
        self.arcs_cost = calculate_arcs_cost()
    def main(self):
        pass


