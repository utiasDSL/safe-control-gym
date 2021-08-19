"""This script runs the Safe Explorer experiment in our Annual Reviews article.

See Figure 7 in https://arxiv.org/pdf/2108.06266.pdf.

"""
import os
import sys
import zipfile
from functools import partial
from itertools import product
from collections import defaultdict
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs, window_func, load_from_log_file, COLORS, LINE_STYLES
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video

# -----------------------------------------------------------------------------------
#                   Plot Funcs
# -----------------------------------------------------------------------------------

def plot_from_exps(legend_dir_specs,
                   out_path="temp.jpg",
                   scalar_name=None,
                   title="Traing Curves",
                   xlabel="Epochs",
                   ylabel="Loss",
                   window=None,
                   x_num_max=None,
                   num_std=1,
                   use_median_quantile=False):
    """Plots 1 stat figure at a time."""
    # Get all stats.
    stats = defaultdict(list)
    for l, dirs in legend_dir_specs.items():
        for d in dirs:
            # Pick from either log source (tensorboard or log text files).
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
        # Use same x for all runs to an algo.
        x = np.array([x[:x_max] for x, _ in runs])[0]

        # Different y for different runs.
        y = np.stack([y[:x_max] for _, y in runs])
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_median = np.median(y, axis=0)
        y_quantiles = np.quantile(y, [0.25, 0.75], axis=0)

        # Record stats.
        processed_stats[name] = {
            "x": x,
            "mean": y_mean,
            "std": y_std,
            "median": y_median,
            "quantiles": y_quantiles,
        }

    # Actual plot.
    fig = plt.figure(figsize=(10, 5))
    plt.clf()
    for i, name in enumerate(sorted(processed_stats.keys())):
        color = COLORS[i]
        x = processed_stats[name]["x"]

        if use_median_quantile:
            y_median = processed_stats[name]["median"]
            y_quantiles = processed_stats[name]["quantiles"]
            y_quant_1st = y_quantiles[0]
            y_quant_3rd = y_quantiles[1]

            plt.plot(x, y_median, label=name, color=color)
            plt.fill_between(x, y_quant_3rd, y_quant_1st, alpha=0.3, color=color)
        else:
            y_mean = processed_stats[name]["mean"]
            y_std = processed_stats[name]["std"]

            plt.plot(x, y_mean, label=name, color=color)
            plt.fill_between(x, y_mean + num_std * y_std, y_mean - num_std * y_std, alpha=0.3, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    return stats, processed_stats


def reward_to_cost(y):
    """Converts RL reward to control cost (for plotting)."""
    return 250 - y


def load_stats(legend_dir_specs, scalar_names=[], window=None, x_num_max=None):
    """Gets all processed stats for multiple scalars."""
    scalar_stats = {}
    for scalar_name in scalar_names:

        # Get all stats.
        stats = defaultdict(list)
        for l, dirs in legend_dir_specs.items():
            for d in dirs:
                # Pick from either log source (tensorboard or log text files).
                path = os.path.join(d, "logs", scalar_name + ".log")
                _, x, _, y = load_from_log_file(path)

                # Smoothing.
                x, y = np.asarray(x), np.asarray(y)
                if "return" in scalar_name.lower() or "reward" in scalar_name.lower():
                    y = reward_to_cost(y)
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
            # Use same x for all runs to an algo.
            x = np.array([x[:x_max] for x, _ in runs])[0]

            # Different y for different runs.
            y = np.stack([y[:x_max] for _, y in runs])
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            y_median = np.median(y, axis=0)
            y_quantiles = np.quantile(y, [0.25, 0.75], axis=0)

            # Record stats.
            processed_stats[name] = {
                "x": x,
                "mean": y_mean,
                "std": y_std,
                "median": y_median,
                "quantiles": y_quantiles,
            }

        # Copy over stats.
        scalar_stats[scalar_name] = deepcopy(processed_stats)

    return scalar_stats


def plot_from_exps2(legend_dir_specs,
                    out_path="temp.jpg",
                    scalar_names=[],
                    title="Traing Curves",
                    sub_titles=["Loss Curve"],
                    xlabel="Epochs",
                    ylabels=["Loss"],
                    window=None,
                    x_num_max=None,
                    num_std=1,
                    use_median_quantile=False,
                    cols_per_row=3):
    """Plots 1 stat figure at a time."""
    # Get all stats.
    scalar_stats = load_stats(legend_dir_specs, scalar_names=scalar_names, window=window, x_num_max=x_num_max)

    # Make plots.
    num_plots = len(scalar_stats)
    num_rows = math.ceil(num_plots / cols_per_row)
    num_cols = min(num_plots, cols_per_row)

    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(nrows=num_rows, ncols=num_cols)

    for idx, scalar_name in enumerate(scalar_names):
        row_idx = idx // num_rows
        col_idx = idx % num_cols
        if num_rows > 1:
            ax = axes[row_idx, col_idx]
        else:
            ax = axes[col_idx]
        processed_stats = scalar_stats[scalar_name]

        for i, name in enumerate(sorted(processed_stats.keys())):
            color_i = (i + 0) % len(COLORS)
            color = COLORS[color_i]

            line_i = (i + 0) % len(LINE_STYLES)
            linestyle = LINE_STYLES[line_i][-1]

            x = processed_stats[name]["x"]

            if use_median_quantile:
                y_median = processed_stats[name]["median"]
                y_quantiles = processed_stats[name]["quantiles"]
                y_quant_1st = y_quantiles[0]
                y_quant_3rd = y_quantiles[1]

                ax.plot(x, y_median, label=name, color=color, alpha=0.7, linestyle=linestyle)
                ax.fill_between(x, y_quant_3rd, y_quant_1st, alpha=0.1, color=color)
            else:
                y_mean = processed_stats[name]["mean"]
                y_std = processed_stats[name]["std"]

                ax.plot(x, y_mean, label=name, color=color, alpha=0.7, linestyle=linestyle)
                ax.fill_between(x, y_mean + num_std * y_std, y_mean - num_std * y_std, alpha=0.1, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[idx])
        ax.set_ylim((-10, None))

    # Postprocess plot.
    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.15)

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
        break
    fig.legend(
        lines,
        labels,
        loc='lower center',
        fancybox=True,
        shadow=True,
        borderaxespad=0.1,
        ncol=7)
    plt.savefig(out_path)
    plt.show()
    return scalar_stats


xabs_legend_map = {
    # baseline
    "ppo": "ppo",
    "ppo_rs_0.15": "ppo_rs_slack0.15",
    "ppo_rs_0.2": "ppo_rs_slack0.2",
    "safe_exp_slack0.15": "se_0.15",
    "safe_exp_slack0.2": "se_0.2",
}

xlh2_legend_map = {
    # baseline
    "ppo": "ppo",
}

thetalh2_legend_map = {
    # baseline
    "ppo": "ppo",
}

legend_maps = {
    "xabs": xabs_legend_map,
    "xlh2": xlh2_legend_map,
    "thetalh2": thetalh2_legend_map,
}

xabs_name_map = {
    # baseline
    "ppo": "PPO",
    # reward shaping
    "ppo_rs_slack0.15": "PPO+cost shaping 0.15",
    "ppo_rs_slack0.2": "PPO+cost shaping 0.2",
    # se
    "se_0.15": "PPO+safety layer 0.15",
    "se_0.2": "PPO+safety layer 0.2",
}


def plot(config):
    """Central plot function."""
    legend_map = legend_maps[config.constraint]

    # Collect results.
    dirs = os.listdir(config.plot_dir)
    # remove pretrain dirs
    dirs = sorted([d for d in dirs if d in legend_map])

    # Make spec.
    spec = {}
    for d in dirs:
        d_name = d.split("/")[-1]
        legend = legend_map[d_name]
        seed_dirs = os.listdir(os.path.join(config.plot_dir, d))
        spec[legend] = [os.path.join(config.plot_dir, d, sd) for sd in seed_dirs]

    # Collective plot (mean only).
    scalar_stats = plot_from_exps2(
        spec,
        out_path=os.path.join(config.plot_dir, "performance.jpg"),
        scalar_names=["stat/ep_return", "stat/constraint_violation"],
        title="Learning Curves",
        sub_titles=["Average Costs", "Total Constraint Violations"],
        xlabel="Step",
        ylabels=["Cost", "Constraint Violations"],
        window=10,
        x_num_max=None,
        num_std=0,
        use_median_quantile=True,
        cols_per_row=3)

    # Save stats to csv.
    curves = ["median", "top_quartile", "bottom_quartile"]
    header = []
    stat_rows = []
    x_already = False

    for scalar_name in scalar_stats:
        stats = scalar_stats[scalar_name]
        true_scalar_name = {"stat/ep_return": "Cost", "stat/constraint_violation": "Constraint Violations"}[scalar_name]

        # Collect stats.
        for algo_name in sorted(stats.keys()):
            true_algo_name = xabs_name_map[algo_name]
            stat = stats[algo_name]
            # X.
            if not x_already:
                header.append("x-Step")
                stat_rows.append(stat["x"])
                x_already = True
            # Y.
            header.extend(["y-{}-{}-{}".format(true_scalar_name, true_algo_name, c) for c in curves])
            stat_rows.append(stat["median"])
            stat_rows.append(stat["quantiles"][1])
            stat_rows.append(stat["quantiles"][0])

    # Make rows.
    stat_mtx = np.array(stat_rows).transpose()
    rows = stat_mtx.tolist()

    # Write to csv.
    csv_path = os.path.join(config.plot_dir, "performance.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("Plotting done.")


# -----------------------------------------------------------------------------------
#                   Main
# -----------------------------------------------------------------------------------

MAIN_FUNCS = {"plot": plot}

if __name__ == "__main__":
    # Make config.
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="plot", help="main function to run.")
    fac.add_argument("--plot_dir", type=str, default="./safe_exp_results")
    fac.add_argument("--constraint", type=str, default="xabs")
    config = fac.merge()
    # Unzip data if the results folder does not exists.
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__))+'/safe_exp_results/'):
        path_to_zip_file = os.path.dirname(os.path.abspath(__file__))+'/safe_exp_results.zip'
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(os.path.abspath(__file__)))
    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))
    func(config)
