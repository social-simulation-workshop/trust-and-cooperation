import argparse
import multiprocessing
import numpy as np
import os
import sys

from utils import Network
from plot import PlotLinesHandler


# parameter
N_PLAYER = 1000
N_NEIGHBOR = 10
EMBEDDEDNESS = 0.67
N_ITERATION = 100000
N_REPLICATION = 10
PAYOFF_X = 1

RNDSEED = 6642
LOG_RECORD_V = 100


def run_simulation(log_data, repli_idx):
    demo = Network(n_player=N_PLAYER, n_neighbor=N_NEIGHBOR, n_iteration=N_ITERATION,
        payoff_x=PAYOFF_X, embeddedness=EMBEDDEDNESS, rnd_seed=RNDSEED+repli_idx)
    demo.simulate(log_verbose_n=10, log_record_v=LOG_RECORD_V)
    log_data.append(demo.get_result())
    plot_result(demo)


def plot_result(demo: Network):
    # plot
    fn_suffix = demo.get_suffix_str()
    
    ## coop
    plot_handler = PlotLinesHandler(xlabel="Time",
                                    ylabel="Relative Frequency",
                                    title=None,
                                    fn="coop",
                                    x_lim=[-1000, 101000], y_lim=[-0.05, 1.05], use_ylim=True,
                                    x_tick=[25000, 100000, 25000], y_tick=[0.0, 1.0, 0.25],
                                    figure_ratio=259/677)
    plot_handler.plot_line(np.array(demo.g1_list), data_log_v=LOG_RECORD_V, linewidth=2, color="black")
    plot_handler.plot_line(np.array(demo.g15_list), data_log_v=LOG_RECORD_V, linewidth=2, color="#656565")
    legend = ["Cooperate with anyone (G_1=1)", "Cooperate with neighbors (G_15=1)"]
    plot_handler.save_fig(legend, fn_suffix=fn_suffix)

    ## trust
    plot_handler = PlotLinesHandler(xlabel="Time",
                                    ylabel="Relative Frequency",
                                    title=None,
                                    fn="trust",
                                    x_lim=[-1000, 101000], y_lim=[-0.05, 1.05], use_ylim=True,
                                    x_tick=[25000, 100000, 25000], y_tick=[0.0, 1.0, 0.25],
                                    figure_ratio=259/677)
    plot_handler.plot_line(np.array(demo.trust_btw_nei), data_log_v=LOG_RECORD_V, linewidth=2, color="#656565")
    plot_handler.plot_line(np.array(demo.trust_btw_stranger), data_log_v=LOG_RECORD_V, linewidth=2, color="black")
    legend = ["Observed trust between neighbors", "Observed trust between strangers"]
    plot_handler.save_fig(legend, fn_suffix=fn_suffix)




if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type=int)
    args = parser.parse_args()
    PAYOFF_X = args.X

    # simulate
    manager = multiprocessing.Manager()
    log_data = manager.list()
    n_cpus = multiprocessing.cpu_count()
    print("cpu count: {}".format(n_cpus))

    args_list = list()
    for repli_idx in range(N_REPLICATION):
        args_list.append([log_data, repli_idx])
    
    pool = multiprocessing.Pool(n_cpus+2)
    pool.starmap(run_simulation, args_list)
    
    # store result
    Network.print_multi_run_result(log_data)

    demo = Network(n_player=N_PLAYER, n_neighbor=N_NEIGHBOR, n_iteration=N_ITERATION,
        payoff_x=PAYOFF_X, embeddedness=EMBEDDEDNESS, rnd_seed=RNDSEED)
    fn = "_".join(["output", demo.get_suffix_str(), "repli_{}".format(N_REPLICATION)]) + ".txt"
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn), 'w')
    sys.stdout = f
    Network.print_multi_run_result(log_data)
    f.close()

    
    

                    

