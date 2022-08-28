"""Experiment with different trajectory similarity metrics.

Todo:
* run the metric test with multiple seeds to check their consistency  
* use a learning-based controller to collect data
* increase computation speed

"""
from multiprocessing.sharedctypes import Value
import os 
from copy import deepcopy
from typing_extensions import final
import torch
import numpy as np
import pickle
import yaml
from functools import partial
import matplotlib.pyplot as plt
from munch import Munch, munchify
import pybullet as p
from termcolor import colored
import tqdm 
import glob 
from collections import defaultdict
import scipy.stats
import csv
from geomloss.kernel_samples import kernel_routines

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import read_file, merge_dict, set_seed_from_config
from safe_control_gym.experiment import Experiment
from safe_control_gym.math_and_models.metrics.similarity_metrics import *


# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------

def add_placeholder_to_config(cfg):
    """Each config should have `algo_config` and `task_config`."""
    if not hasattr(cfg, "task_config"):
        cfg.task_config = munchify({})
    if not hasattr(cfg, "algo_config"):
        cfg.algo_config = munchify({})


def get_variant_config(v_config, config): 
    """Combines the 2 configs to get final variant config, 
        contains `task_config`, `algo_config`, `task`, `algo`, `restore`, `checkpoint`. 
    """
    final_config = munchify({})
    final_config.task_config = deepcopy(config.task_config)
    final_config.algo_config = deepcopy(config.algo_config)
    for k in ["task", "algo", "restore", "checkpoint"]:
        if hasattr(config, k):
            final_config[k] = config[k]
    merge_dict(final_config, v_config)
    return final_config


def make_experiment(config, seed):
    """Instantiates an experiment with an env and a ctrl."""
    # env
    env_func = partial(make, config.task, **config.task_config)
    env = env_func()
    env.seed(seed)
    # ctrl 
    ctrl = make(config.algo, env_func, seed=seed, **config.algo_config)
    ctrl.reset()
    if config.restore:
        ctrl.load(os.path.join(config.restore, config.checkpoint))
    # experiment
    experiment = Experiment(env, ctrl)
    return experiment


def get_kernel_cost_func(name, loss_func_name, cost_or_kernel="cost"):
    """"""
    if cost_or_kernel not in ["cost", "kernel"]:
        raise ValueError("`cost_or_kernel` must be `cost` or `kernel` only.")
    if loss_func_name == "sinkhorn":
        if cost_or_kernel == "kernel":
            return None
        # sinkhorn in geomloss either uses a callable, or delegate to a cost func map keyed by value of `p`
        # reference: https://github.com/jeanfeydy/geomloss/blob/3ed5c8504b9d1b2266ae9325716341f272afce11/geomloss/sinkhorn_samples.py#L172
        if name in globals():
            return eval(name)
        else:
            return None
    elif loss_func_name == "hausdorff":
        if cost_or_kernel == "cost":
            return None
        # hausdorff in geomloss either uses a callable, or delegate to a kernel func map keyed by value of `name`, 
        # but `name` is not exposed in geomloss.SamplesLoss, so we do it here instead
        # https://github.com/jeanfeydy/geomloss/blob/3ed5c8504b9d1b2266ae9325716341f272afce11/geomloss/kernel_samples.py#L107
        if name in globals():
            return eval(name)
        elif name in kernel_routines:
            return kernel_routines[name]
        else:
            raise ValueError("The given kernel name is not available in safe-control-gym or geomloss's kernel_routines.")
    else:
        # for MMD, not needed yet.
        return None
    


# -----------------------------------------------------------------------------------
#                   Test
# -----------------------------------------------------------------------------------
    
def test_trajectory_metric(config):
    """Compute the similarity metric, performance metric and their correlations with
        varying env params, ctrl params, priors, and etc.
    """
    # params 
    n_episodes = config.n_episodes
    n_variants = len(config.variant_config)
    env_seed_offset = 100
    
    # config setup 
    add_placeholder_to_config(config)
    set_seed_from_config(config)

    # base/control group.
    experiment = make_experiment(config, config.seed)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, verbose=False)
    experiment.close()
    
    # variants/experimental group.
    v_trajs_data,v_metrics = [], []
    for i, v_config in enumerate(config.variant_config):
        # variant setup 
        if config.metric in ["lsed", "dtw", "edr", "lcss", "discrete_frechet"]:
            # requires the same initial conditions btw reference traj and variant traj
            v_seed = config.seed
        elif config.metric in ["mmd_loss", "geom_loss"]:
            # only require random samples from reference traj and variant traj
            # no need for matching initial conditions, here we use a different seed for each variant
            v_seed = config.seed + env_seed_offset + i
        else:
            raise NotImplementedError("The given similarity metric is not available for data collection.")
        final_v_config = get_variant_config(v_config, config)

        experiment = make_experiment(final_v_config, v_seed)
        trajs_data_, metrics_ = experiment.run_evaluation(n_episodes=n_episodes, verbose=False)
        experiment.close()
        
        v_trajs_data.append(trajs_data_)
        v_metrics.append(metrics_)
        
    # similarity metrics 
    if config.metric in ["lsed", "dtw", "edr", "lcss", "discrete_frechet"]: 
        # compute metric of each pair of ref traj and variant traj, use the average as final metric value 
        # construct metric function
        metric_func = eval(config.metric)
        metric_kwargs = {
            "distance_func": state_distance,
            "distance_func_kwargs": {
                "weights": None, 
                "rot_mode": "geodesic", 
                "task": config.task, 
                "quad_type": config.task_config.get("quad_type", None),
            }
        }
        # compute metric values
        metric_vals = np.zeros(n_variants)
        for i in tqdm.tqdm(range(n_variants)):
            mval = 0.
            for j in range(n_episodes):
                mval += metric_func(trajs_data["obs"][j], 
                                    v_trajs_data[i]["obs"][j], 
                                    **metric_kwargs)
            metric_vals[i] = mval / n_episodes
            print(colored("param: {} | metric: {}".format(config.variants[i], metric_vals[i]), "blue"))     
    elif config.metric in ["mmd_loss", "geom_loss"]:
        # compute metric btw batch of ref traj samples and variant traj samples
        # construct metric function
        metric_func = eval(config.metric)
        if config.metric == "mmd_loss":
            metric_kwargs = {
                "mode": config.mmd_mode,
                "sigma": config.mmd_sigma,
            }
        elif config.metric == "geom_loss":
            # sinkhorn loss with small blur can approximate Wasserstein distance
            # reference: http://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html
            metric_kwargs = {
                "loss": config.geom_loss_func,
                "p": config.geom_loss_p,
                "blur": config.geom_loss_blur,
                # to swap in a custom state distance function if needed, 
                # also to fix the bug in geomloss for `hausdorff` (see `get_kernel_cost_func`) 
                "cost": get_kernel_cost_func(
                    config.get("geom_loss_cost", None), 
                    config.geom_loss_func, 
                    cost_or_kernel="cost"),
                "kernel": get_kernel_cost_func(
                    config.get("geom_loss_kernel", None), 
                    config.geom_loss_func, 
                    cost_or_kernel="kernel"),
            }
        else:
            raise NotImplementedError("The given distribution metric is not available.")
        # compute metric values
        metric_vals = np.zeros(n_variants)
        processed_data = encode_data(trajs_data, config.tuple_length)
        for i in tqdm.tqdm(range(n_variants)):
            processed_v_data = encode_data(v_trajs_data[i], config.tuple_length)
            metric_vals[i] = metric_func(processed_data, 
                                         processed_v_data, 
                                         **metric_kwargs)
            print(colored("param: {} | metric: {}".format(config.variants[i], metric_vals[i]), "blue"))     
    else:
        raise NotImplementedError("The given similarity metric is not available for metric computation.")
    
    # performance metrics
    rmse_vals = np.asarray([vm["average_rmse"] for vm in v_metrics])
    
    # calculate correlation btw similarity metric and performance metric (RMSE)
    # can be seen as cross-correlation of two series as a function of trajectories
    # second element of the correlation metric output is the p-value
    correlation_metrics = {
        # Pearson's rho, product-moment correlation coeff
        "pearsonr": float(scipy.stats.pearsonr(metric_vals, rmse_vals)[0]),
        # Spearman's rho, rank correlation coeff
        "spearmanr": float(scipy.stats.spearmanr(metric_vals, rmse_vals)[0]),
        # Kendall's tau, rank correlation coeff
        "kendalltau": float(scipy.stats.kendalltau(metric_vals, rmse_vals)[0]),
    }
        
    # save to file 
    os.makedirs(config.eval_output_dir, exist_ok=True)
    file_name = os.path.join(config.eval_output_dir, config.file_name)
    # convert performance metrics to float to save in yaml
    processed_v_metrics = []
    for i in range(len(v_metrics)):
        processed_ = {}
        for k, v in v_metrics[i].items():
            processed_[k] = float(v)
        processed_v_metrics.append(processed_)
    results = {
        "reference": config.reference,
        "variants": config.variants,
        "xlabel": config.xlabel,
        "metric_vals": metric_vals.tolist(),
        "rmse_vals": rmse_vals.tolist(),
        "performance_metrics": processed_v_metrics,
        "correlation_metrics": correlation_metrics,
    }
    with open(file_name, "w")as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # plot 
    if config.plot:
        fig = plt.figure(figsize=(10, 3))
        axes = fig.subplots(nrows=1, ncols=2)
        # metric plot  
        axes[0].plot(config.variants, metric_vals)
        axes[0].axvline(x=float(config.reference), linewidth=1, color='r', linestyle="--", label="reference value")
        axes[0].set_xlabel(config.xlabel)
        axes[0].set_ylabel("{} distance to reference".format(config.metric))
        axes[0].set_title("{}".format(config.metric))
        # rmse plot
        axes[1].plot(config.variants, rmse_vals)
        axes[1].axvline(x=float(config.reference), linewidth=1, color='r', linestyle="--", label="reference value")
        axes[1].set_xlabel(config.xlabel)
        axes[1].set_ylabel("average RMSE")
        axes[1].set_title("RMSE")
        
        fig_name = os.path.join(config.eval_output_dir, config.fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.show()
    
    print("test done ...")    
    
    
    
# -----------------------------------------------------------------------------------
#                   Plot
# -----------------------------------------------------------------------------------
    
def plot_metric(config):
    """Shows the similarity & performance metric v.s. varying factor side-by-side, 
        also computes their (average) correlation.
    """
    # params 
    xlabel, reference, variants = None, None, None
    
    # load results 
    algo_seed_dirs_pat = os.path.join(config.eval_output_dir, "seed*")        
    algo_seed_dirs = glob.glob(algo_seed_dirs_pat)
    seed_data = defaultdict(list)
    
    for seed_dir in algo_seed_dirs:
        eval_file = os.path.join(seed_dir, config.file_name)
        print(eval_file)
        result = read_file(eval_file)
        # load results 
        if not xlabel:
            xlabel, reference, variants = result["xlabel"], result["reference"], result["variants"]
        # shape (n_episodes,)
        rmse = [res["average_rmse"] for res in result["performance_metrics"]]
        metric_vals = result["metric_vals"]
        seed_data["rmse"].append(rmse)
        seed_data["metric_vals"].append(metric_vals)
        for k, v in result["correlation_metrics"].items():
            seed_data["correlation_{}".format(k)].append(v)
            
    # each is shape (n_seeds, n_episodes)
    for scalar_name, scalar_seed_data in seed_data.items():
        seed_data[scalar_name] = np.asarray(scalar_seed_data)
            
    # save to file
    os.makedirs(config.eval_output_dir, exist_ok=True)
    file_name = os.path.join(config.eval_output_dir, config.file_name)
    results = {
        "variants": variants,
        "metric": seed_data["metric_vals"].tolist(),
        "metric_mean": seed_data["metric_vals"].mean(0).tolist(),
        "metric_std": seed_data["metric_vals"].std(0).tolist(),
        "rmse": seed_data["rmse"].tolist(),
        "rmse_mean": seed_data["rmse"].mean(0).tolist(),
        "rmse_std": seed_data["rmse"].std(0).tolist(),
    }
    for k, v in seed_data.items():
        if k.startswith("correlation_"):
            results[k] = v.tolist()
            results["{}_mean".format(k)] = float(v.mean())
            results["{}_std".format(k)] = float(v.std())
    with open(file_name, "w")as f:
        yaml.dump(results, f, default_flow_style=False)

    # plot 
    fig = plt.figure(figsize=(10, 3))
    axes = fig.subplots(nrows=1, ncols=2)

    # metric plot 
    metric_mean = seed_data["metric_vals"].mean(0)
    metric_std = seed_data["metric_vals"].std(0)
    axes[0].plot(variants, metric_mean, alpha=0.7)
    axes[0].fill_between(variants, metric_mean+metric_std, metric_mean-metric_std, alpha=0.3)
    axes[0].axvline(x=float(reference), linewidth=1, color='r', linestyle="--", label="reference value")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("{} distance to reference".format(config.metric))
    axes[0].set_title("{}".format(config.metric))

    # rmse plot
    rmse_mean = seed_data["rmse"].mean(0)
    rmse_std = seed_data["rmse"].std(0)
    axes[1].plot(variants, rmse_mean, alpha=0.7)
    axes[1].fill_between(variants, rmse_mean+rmse_std, rmse_mean-rmse_std, alpha=0.3)
    axes[1].axvline(x=float(reference), linewidth=1, color='r', linestyle="--", label="reference value")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("average RMSE")
    axes[1].set_title("RMSE")
        
    fig_name = os.path.join(config.eval_output_dir, config.fig_name)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
    
    print("plot done ...") 


def plot_correlation(config):
    """Compile the calculated correlation coeffs of similarity metric and performance metric 
        into a table, save as a csv.
    """
    legend_map = {
        "lsed": "LSED",
        "dtw": "DTW",
        "edr": "EDR",
        "lcss": "LCSS",
        "discrete_frechet": "Frechet",
        "mmd_loss": "MMD",
        "wasserstein": "Wasserstein",
        "hausdorff": "Hausdorff",
    }
    correlation_metric_order = [
        "pearsonr", "spearmanr", "kendalltau"
    ]
    legend_order = [
        "LSED", "DTW", "EDR", "LCSS", "Frechet", 
        "MMD", "Wasserstein", "Hausdorff"
    ]
    
    # load results 
    correlation_data = defaultdict(lambda: defaultdict(list))
    for d, legend in legend_map.items():
        algo_seed_dirs_pat = os.path.join(config.eval_output_dir, d, "seed*")        
        algo_seed_dirs = glob.glob(algo_seed_dirs_pat)

        for seed_dir in algo_seed_dirs:
            eval_file = os.path.join(seed_dir, config.file_name)
            print(eval_file)
            result = read_file(eval_file)
            for k, v in result["correlation_metrics"]:
                correlation_data[legend][k].append(v)
    
    # save to csv
    csv_mtx = np.zeros((len(legend_order), len(correlation_metric_order)))
    for i, legend in enumerate(legend_order):
        for j, k in enumerate(correlation_metric_order):
            cor_metric = np.asarray(correlation_data[legend][k])
            csv_mtx[i][j] = cor_metric.mean()
    csv_rows = csv_mtx.tolist()
    csv_path = os.path.join(config.eval_output_dir, config.csv_file_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(correlation_metric_order)
        writer.writerows(csv_rows)

    print("plot done ...")
    
    
    
    
# -----------------------------------------------------------------------------------
#                   Main
# -----------------------------------------------------------------------------------
    
if __name__ == "__main__":
    # Make config.
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="train", help="main function to run.")
    fac.add_argument("--thread", type=int, default=0, help="number of threads to use (set by torch).")
    fac.add_argument("--render", action="store_true", help="if to render in policy test.")
    fac.add_argument("--verbose", action="store_true", help="if to print states & actions in policy test.")
    fac.add_argument("--use_adv", action="store_true", help="if to evaluate against adversary.")
    fac.add_argument("--set_test_seed", action="store_true", help="if to set seed when testing policy.")
    fac.add_argument("--eval_output_dir", type=str, default="", help="folder path to save evaluation results.")
    fac.add_argument("--eval_output_path", type=str, default="test_results.pkl", help="file path to save evaluation results.")

    # cutom args
    fac.add_argument("--checkpoint", type=str, default="model_latest.pt", help="file path to model checkpoint to load.")
    fac.add_argument("--n_episodes", type=int, default=10, help="number of test episodes.")
    fac.add_argument("--metric", type=str, default="mse", help="metric name.")
    fac.add_argument("--fig_name", type=str, default="temp.png", help="fig name to save.")
    fac.add_argument("--file_name", type=str, default="temp.yaml", help="file name to save.")
    fac.add_argument("--tuple_length", type=int, default=1)
    fac.add_argument("--mmd_mode", type=str, default="gaussian")
    fac.add_argument("--mmd_sigma", type=float, default=10.0)
    fac.add_argument("--geom_loss_func", type=str, default="sinkhorn")
    fac.add_argument("--geom_loss_p", type=int, default=2)
    fac.add_argument("--geom_loss_blur", type=float, default=0.05)
    fac.add_argument("--geom_loss_cost", type=str, help="distance/cost function between 2 samples to use for `sinkhorn` loss.")
    fac.add_argument("--geom_loss_kernel", type=str, help="same as argument `cost`, but for `hausdorff` loss.")
    fac.add_argument("--plot", action="store_true", help="if to plot the results.")
    fac.add_argument("--csv_file_name", type=str, default="temp.csv", help="csv file name to save.")

    config = fac.merge()
    # System settings.
    if config.thread > 0:
        # E.g. set single thread for less context switching
        torch.set_num_threads(config.thread)
    # Execute.
    func = getattr(config, "func", None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))
    eval(func)(config)
