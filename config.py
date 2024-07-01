import argparse


parser = argparse.ArgumentParser()


def test_case():
    parser.add_argument('--exp_name', type=str, default='test_case')
    parser.add_argument('--n_trains', type=int, default=2)
    parser.add_argument('--n_stations', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--block_station_link', type=list, default=[[0, 1],
                                                                    [1, 2],
                                                                    [2, 3]])
    parser.add_argument('--time_zone', type=int, default=50)
    parser.add_argument('--train_type', type=list, default=[0, 0])
    parser.add_argument('--train_speed', type=list, default=[250, 300])
    parser.add_argument('--direction', type=list, default=[0, 0, 0])
    parser.add_argument('--distance', type=list, default=[160, 200, 250])
    parser.add_argument('--down_run_time', type=list, default=[[3, 4, 5]])
    parser.add_argument('--up_run_time', type=list, default=[[3, 4, 5]])
    parser.add_argument('--start_time', type=list, default=[0, 10])
    parser.add_argument('--shift_scope', type=list, default=[-1, 1])
    parser.add_argument('--stop_plan', type=list, default=[[0, 0]])
    parser.add_argument('--punish', type=int, default=10e5)
    parser.add_argument('--dep_track_headway', type=int, default=2)
    parser.add_argument('--arr_track_headway', type=int, default=2)
    parser.add_argument('--time_loss_ac', type=int, default=1)
    parser.add_argument('--time_loss_de', type=int, default=2)
    parser.add_argument('--min_dwell_time', type=int, default=2)
    parser.add_argument('--max_dwell_time', type=int, default=3)
    return parser.parse_args()


def dl_line_3_4():
    parser.add_argument('--exp_name', type=str, default='dl_line_3_4')
    parser.add_argument('--n_trains', type=int, default=3)
    parser.add_argument('--n_stations', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--time_zone', type=int, default=300)
    parser.add_argument('--train_type', type=list, default=[0, 1, 1])
    parser.add_argument('--train_speed', type=list, default=[250, 300])
    parser.add_argument('--direction', type=list, default=[0, 0, 0])
    parser.add_argument('--distance', type=list, default=[160, 200, 180])
    parser.add_argument('--down_run_time', type=list, default=[[38, 48, 42],
                                                               [32, 40, 36]])
    parser.add_argument('--up_run_time', type=list, default=[[38, 48, 42],
                                                             [32, 40, 36]])
    parser.add_argument('--start_time', type=list, default=[0, 50, 100])
    parser.add_argument('--shift_scope', type=list, default=[-5, 5])
    parser.add_argument('--stop_plan', type=list, default=[[1, 1, 0],
                                                           [0, 0, 0]])
    parser.add_argument('--punish', type=int, default=10e5)
    parser.add_argument('--dep_track_headway', type=int, default=4)
    parser.add_argument('--arr_track_headway', type=int, default=4)
    parser.add_argument('--time_loss_ac', type=int, default=2)
    parser.add_argument('--time_loss_de', type=int, default=3)
    parser.add_argument('--min_dwell_time', type=int, default=2)
    parser.add_argument('--max_dwell_time', type=int, default=3)
    return parser.parse_args()


def dl_line_3_4():
    parser.add_argument('--exp_name', type=str, default='dl_line_3_4')
    parser.add_argument('--n_trains', type=int, default=3)
    parser.add_argument('--n_stations', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--time_zone', type=int, default=300)
    parser.add_argument('--train_type', type=list, default=[0, 1, 1])
    parser.add_argument('--train_speed', type=list, default=[250, 300])
    parser.add_argument('--direction', type=list, default=[0, 0, 0])
    parser.add_argument('--distance', type=list, default=[160, 200, 180])
    parser.add_argument('--down_run_time', type=list, default=[[38, 48, 42],
                                                               [32, 40, 36]])
    parser.add_argument('--up_run_time', type=list, default=[[38, 48, 42],
                                                             [32, 40, 36]])
    parser.add_argument('--start_time', type=list, default=[0, 50, 100])
    parser.add_argument('--shift_scope', type=list, default=[-5, 5])
    parser.add_argument('--stop_plan', type=list, default=[[1, 1, 0],
                                                           [0, 0, 0]])
    parser.add_argument('--punish', type=int, default=10e5)
    parser.add_argument('--dep_track_headway', type=int, default=4)
    parser.add_argument('--arr_track_headway', type=int, default=4)
    parser.add_argument('--time_loss_ac', type=int, default=2)
    parser.add_argument('--time_loss_de', type=int, default=3)
    parser.add_argument('--min_dwell_time', type=int, default=2)
    parser.add_argument('--max_dwell_time', type=int, default=3)
    return parser.parse_args()


def sl_line_3_4():
    parser.add_argument('--exp_name', type=str, default='sl_line_3_4')
    parser.add_argument('--n_trains', type=int, default=3)
    parser.add_argument('--n_stations', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--time_zone', type=int, default=60)
    # parser.add_argument('--train_type', type=list, default=[0, 0, 0])
    # parser.add_argument('--train_speed', type=list, default=[250, 300])
    parser.add_argument('--direction', type=list, default=[0, 0, 1])
    parser.add_argument('--distance', type=list, default=[8.1, 10.2, 9])
    parser.add_argument('--down_run_time', type=list, default=[9, 8, 7])
    parser.add_argument('--up_run_time', type=list, default=[9, 8, 7])
    parser.add_argument('--start_time', type=list, default=[0, 17, 47])
    # parser.add_argument('--shift_scope', type=list, default=[-4, 4])
    parser.add_argument('--stop_plan', type=list, default=[[0, 0, 0],
                                                           [0, 0, 0]])
    parser.add_argument('--punish', type=int, default=10e5)
    parser.add_argument('--consecutive_headway', type=int, default=4)
    parser.add_argument('--station_headway', type=int, default=2)
    parser.add_argument('--block_headway', type=int, default=2)
    parser.add_argument('--time_loss_ac', type=int, default=2)
    parser.add_argument('--time_loss_de', type=int, default=3)
    parser.add_argument('--min_dwell_time', type=int, default=4)
    parser.add_argument('--max_dwell_time', type=int, default=25)
    return parser.parse_args()
