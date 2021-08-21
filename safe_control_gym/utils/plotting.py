"""Plotting utilities.

"""
from collections import defaultdict
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DIV_LINE_WIDTH = 50


COLORS = [
    "blue",
    "green",
    "red",
    "black",
    "cyan",
    "magenta",
    "yellow",
    "brown",
    "purple",
    "pink",
    "orange",
    "teal",
    "coral",
    "lightblue",
    "lime",
    "lavender",
    "turquoise",
    "darkgreen",
    "tan",
    "salmon",
    "gold",
    "lightpurple",
    "darkred",
    "darkblue",
]


LINE_STYLES = [
    ('solid', 'solid'),
    ('dotted', 'dotted'),
    ('dashed', 'dashed'),
    ('dashdot', 'dashdot'),
]


LINE_STYLES2 = [('loosely dotted', (0, (1, 10))), ('dotted', (0, (1, 1))),
                ('densely dotted', (0, (1, 1))),
                ('loosely dashed', (0, (5, 10))), ('dashed', (0, (5, 5))),
                ('densely dashed', (0, (5, 1))),
                ('loosely dashdotted', (0, (3, 10, 1, 10))),
                ('dashdotted', (0, (3, 5, 1, 5))),
                ('densely dashdotted', (0, (3, 1, 1, 1))),
                ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
                ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
]


def rolling_window(a, window):
    """Window data.

    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func):
    """Evaluate a function on windowed data.

    """
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window - 1:], yw_func


def filter_log_dirs(pattern, negative_pattern=" ", root="./log", **kwargs):
    """Gets list of experiment folders as specified.

    """
    dirs = [item[0] for item in os.walk(root)]
    leaf_dirs = []
    for i in range(len(dirs)):
        if i + 1 < len(dirs) and dirs[i + 1].startswith(dirs[i]):
            continue
        leaf_dirs.append(dirs[i])
    names = []
    p = re.compile(pattern)
    np = re.compile(negative_pattern)
    for dir in leaf_dirs:
        if p.match(dir) and not np.match(dir):
            names.append(dir)
            print(dir)
    print("")
    return sorted(names)


def align_runs(xy_list, x_num_max=None):
    """Aligns the max of the x data across runs.

    """
    x_max = float("inf")
    for x, y in xy_list:
        # Align length of x data (get min across all runs).
        x_max = min(x_max, len(x))
    if x_num_max:
        x_max = min(x_max, x_num_max)
    aligned_list = [[x[:x_max], y[:x_max]] for x, y in xy_list]
    return aligned_list


def smooth_runs(xy_list, window=10):
    """Smooth the data curves by mean filtering.

    """
    smoothed_list = [
        window_func(np.asarray(x), np.asarray(y), window, np.mean)
        for x, y in xy_list
    ]
    return smoothed_list


def select_runs(xy_list, criterion, top_k=0):
    """Pickes the top k runs based on a criterion.

    """
    perf = [criterion(y) for _, y in xy_list]
    top_k_runs = np.argsort(perf)[-top_k:]
    selected_list = []
    for r, (x, y) in enumerate(xy_list):
        if r in top_k_runs:
            selected_list.append((x, y))
    return selected_list


def interpolate_runs(xy_list, interp_interval=100):
    """Uses the same x data by interpolation across runs.

    """
    x_right = float("inf")
    for x, y in xy_list:
        x_right = min(x_right, x[-1])
    # Shape: (data_len,).
    x = np.arange(0, x_right, interp_interval)
    y = []
    for x_, y_ in xy_list:
        y.append(np.interp(x, x_, y_))
    # Shape: (num_runs, data_len).
    y = np.asarray(y)
    return x, y


def load_from_log_file(path):
    """Return x, y sequence data from the stat csv.

    """
    with open(path, "r") as f:
        lines = f.readlines()
    # Labels.
    xk, yk = [k.strip() for k in lines[0].strip().split(",")]
    # Values.
    x, y = [], []
    for l in lines[1:]:
        data = l.strip().split(",")
        x.append(float(data[0].strip()))
        y.append(float(data[1].strip()))
    x = np.array(x)
    y = np.array(y)
    return xk, x, yk, y


def load_from_logs(log_dir):
    """Return dict of stats under log_dir folder (`exp_dir/logs/`).

    """
    log_files = []
    # Fetch all log files.
    for r, _, f in os.walk(log_dir):
        for file in f:
            if ".log" in file:
                log_files.append(os.path.join(r, file))
    # Fetch all stats from log files.
    data = {}
    for path in log_files:
        name = path.split(log_dir)[-1].replace(".log", "")
        xk, x, yk, y = load_from_log_file(path)
        data[name] = (xk, x, yk, y)
    return data


def plot_from_logs(src_dir, out_dir, window=None, keys=None):
    """Generate a plot for each stat in an experiment `logs` folder.
    
    Args:
        src_dir (str): folder to read logs.
        out_dir (str): folder to save figures.
        window (int): window size for smoothing.
        keys (list): specify name of stats to plot, None means plot all.

    """
    # Find all logs.
    log_files = []
    for r, _, f in os.walk(src_dir):
        for file in f:
            if ".log" in file:
                log_files.append(os.path.join(r, file))
    # Make a figure for each log file.
    stats = {}
    for path in log_files:
        name = path.split(src_dir)[-1].replace(".log", "")
        if keys:
            if name not in keys:
                continue
        xk, x, yk, y = load_from_log_file(path)
        stats[name] = (xk, x, yk, y)
        if window:
            x, y = window_func(x, y, window, np.mean)
        plt.clf()
        plt.plot(x, y)
        plt.title(name)
        plt.xlabel(xk)
        plt.ylabel(yk)
        plt.savefig(os.path.join(out_dir, name.replace("/", "-") + ".jpg"))
    return stats


def plot_from_tensorboard_log(src_dir,
                              out_dir,
                              window=None,
                              keys=None,
                              xlabel="step"):
    """Generates a plot for each stat from tfb log file in source folder.
    
    """
    event_acc = EventAccumulator(src_dir)
    event_acc.Reload()
    if not keys:
        keys = event_acc.Tags()["scalars"]
    stats = {}
    for k in keys:
        _, x, y = zip(*event_acc.Scalars(k))
        x, y = np.asarray(x), np.asarray(y)
        stats[k] = (x, y)
        if window:
            x, y = window_func(x, y, window, np.mean)
        plt.clf()
        plt.plot(x, y)
        plt.title(k)
        plt.xlabel(xlabel)
        plt.ylabel(k)
        # Use "-" instead of "/" to connect group and stat name.
        out_path = os.path.join(out_dir, k.replace("/", "-") + ".jpg")
        plt.savefig(out_path)
    return stats


def plot_from_experiments(legend_dir_specs,
                          out_path="temp.jpg",
                          scalar_name=None,
                          title="Traing Curves",
                          xlabel="Epochs",
                          ylabel="Loss",
                          window=None,
                          x_num_max=None,
                          num_std=1,
                          use_tb_log=True
                          ):
    """Generates plot among algos, each with several seed runs.
    
    Example: 
        make a plot on average reward for gnn and mlp:: 
        
        > plot_from_experiments(
            {
                "gnn": [
                    "results/algo1/seed0", 
                    "results/algo1/seed1", 
                    "results/algo1/seed2"
                ],
                "mlp": [
                    "results/algo2/seed6",
                    "results/algo2/seed1",
                    "results/algo2/seed9",
                    "results/algo2/seed3"
                ],
            },
            out_path="avg_reward.jpg",
            scalar_name="loss_eval/total_rewards",
            title="Average Reward",
            xlabel="Epochs",
            ylabel="Reward",
            window=10
        )

    """
    assert scalar_name is not None, "Must provide a scalar name to plot"
    # Get all stats.
    stats = defaultdict(list)
    for l, dirs in legend_dir_specs.items():
        for d in dirs:
            # Pick from either log source (tensorboard or log text files).
            if use_tb_log:
                event_acc = EventAccumulator(d)
                event_acc.Reload()
                _, x, y = zip(*event_acc.Scalars(scalar_name))
                del event_acc
            else:
                path = os.path.join(d, "logs", scalar_name + ".log")
                _, x, _, y = load_from_log_file(path)
            # Smoothing.
            x, y = np.asarray(x), np.asarray(y)
            if window:
                x, y = window_func(x, y, window, np.mean)
            stats[l].append([x, y])
    # Post-processing.
    x_max = float("inf")
    for _, runs in stats.items():
        for x, y in runs:
            # Align length of x data (get min across all runs & all algos).
            x_max = min(x_max, len(x))
    if x_num_max:
        x_max = min(x_max, x_num_max)
    processed_stats = {}
    for name, runs in stats.items():
        # Use same x for all runs to an algorithm.
        x = np.array([x[:x_max] for x, _ in runs])[0]
        # Different y for different runs.
        y = np.stack([y[:x_max] for _, y in runs])
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        processed_stats[name] = [x, y_mean, y_std]
    # Actual plot.
    plt.clf()
    for i, name in enumerate(processed_stats.keys()):
        color = COLORS[i]
        x, y_mean, y_std = processed_stats[name]
        plt.plot(x, y_mean, label=name, color=color)
        plt.fill_between(x,
                         y_mean + num_std * y_std,
                         y_mean - num_std * y_std,
                         alpha=0.3,
                         color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    return stats, processed_stats


def get_log_dirs(all_logdirs,
                 select=None,
                 exclude=None
                 ):
    """Find all folders for plotting.

    All 3 arguments can be exposed as list args from command line.

    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 
        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.

    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    # Enforce selection rules, which check logdirs for certain substrings. Makes it easier to look
    # at graphs from particular ablations, if you launch many jobs at once with similar names.
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [
            log for log in logdirs if all(not (x in log) for x in exclude)
        ]
    # Verify logdirs.
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)
    return logdirs
