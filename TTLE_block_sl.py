import numpy as np
import math as m
import pdb
import random


class ENVIRONMENT:
    def __init__(self, config):
        # loading train information
        self.n_train = config.n_train
        self.n_down_train = config.n_down_train
        self.n_up_train = config.n_up_train
        # loading infrastructure information
        self.n_block = config.n_block
        self.n_station = config.n_station
        # loading environment parameters
        self.time_loss_acceleration = config.time_loss_acceleration
        self.time_loss_deceleration = config.time_loss_deceleration
        self.station_headway = config.station_headway
        self.block_headway = config.block_headway
        # when the latter train stops at the tail station in the present block.
        self.consecutive_headway_stop = config.consecutive_headway_stop
        # when the latter train passes through the tail station in the present block.
        self.consecutive_headway_pass = config.consecutive_headway_pass
        self.down_run_time = config.down_run_time
        self.up_run_time = config.up_run_time
        self.distance = config.distance
        self.punish = config.punish
        self.action_set = config.action_set
        self.time_zone = config.time_zone
        # not check constraints if test mode
        self.is_test_mode = config.is_test_mode


