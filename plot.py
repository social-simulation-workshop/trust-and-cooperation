import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from utils import eprint

sns.set()
sns.set_style("white")
# sns.set_style("whitegrid")


class PlotLinesHandler:
    _ids = itertools.count(0)
    EPSILON = 10**-5

    def __init__(self, title, xlabel, ylabel, fn,
        x_lim, y_lim, x_tick, y_tick, figure_ratio, use_ylim=True, figure_size=12, x_as_kilo=False,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgfiles")) -> None:
        
        self.id = next(self._ids)

        self.output_dir = output_dir
        self.fn = fn
        self.legend_list = list()

        plt.figure(self.id, figsize=(figure_size, figure_size*figure_ratio), dpi=160)
        if title is not None:
            plt.title(title, fontweight="bold")
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        ax = plt.gca()
        if x_lim is not None:
            ax.set_xlim([x_lim[0], x_lim[1]])
            if x_as_kilo:
                x_tick_label = ["{}K".format(int(i/1000)) for i in np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2])]
                plt.xticks(np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2]), x_tick_label)
            else:
                plt.xticks(np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2]))
        if y_lim is not None and use_ylim:
            ax.set_ylim([y_lim[0], y_lim[1]])
            plt.yticks(np.arange(y_tick[0], y_tick[1]+self.EPSILON, step=y_tick[2]))
        self.use_ylim = use_ylim
            

    def plot_line(self, data, data_log_v=1, linewidth=1, color="", alpha=1.0):
        plt.figure(self.id)
        if color:
            plt.plot((np.arange(data.shape[-1])+1)*data_log_v, data,
                linewidth=linewidth, color=color, alpha=alpha)
        else:
            plt.plot((np.arange(data.shape[-1])+1)*data_log_v, data,
                linewidth=linewidth)

    def save_fig(self, legend=[], fn_prefix="", fn_suffix=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.figure(self.id)
        if not self.use_ylim:
            fn_suffix += "y_unlimited"
        fn = "_".join([fn_prefix, self.fn, fn_suffix]).strip("_") + ".png"
        
        if legend:
            plt.legend(legend)
        # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        plt.savefig(os.path.join(self.output_dir, fn))
        eprint("fig save to {}".format(os.path.join(self.output_dir, fn)))