"""Experiment with different trajectory similarity metrics.

Todo:
* run the metric test with multiple seeds to check their consistency  
* use a learning-based controller to collect data
* increase computation speed

"""
import os 
from copy import deepcopy
import torch
import numpy as np
import pickle
import yaml
import json
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
import subprocess
import shlex 
import itertools
import h5py 

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import read_file, merge_dict, set_seed_from_config
from safe_control_gym.experiment import Experiment
from safe_control_gym.explore_experiment import ExploreExperiment
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
    try:
        ctrl = make(config.algo, 
                    env_func, 
                    seed=seed,
                    training=False, 
                    **config.algo_config)
    except:
        # so non-learning controller has bug in sending `training=False` as init arg
        ctrl = make(config.algo, 
                    env_func, 
                    seed=seed,
                    **config.algo_config)
    ctrl.reset()
    if config.restore:
        ctrl.load(os.path.join(config.restore, config.checkpoint))
    # explore strategy 
    explore_policy = None 
    if hasattr(config, "explore") and hasattr(config, "explore_config"):
        explore_policy = make(config.explore, **config.explore_config)
    # experiment
    # experiment = Experiment(env, ctrl)
    experiment = ExploreExperiment(env, ctrl, explore_policy=explore_policy)
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
    
    
def compute_metric(config, trajs_data, v_trajs_data):
    """Gets the traj similarity metric values among list of variants.
    
    Args:
        config (Munch dict): configurations 
        trajs_data (dict|list): if dict, contains reference traj data, trajs_data["obs"] is n_episode of traj observations;
            if list, similar to v_trajs_data below.
        v_trajs_data (list): list of dict, each dict contains traj data, v_trajs_data[0]["obs"] is n_episode of traj observations.
        
    Returns:
        ndarray: metric values of shape (n_variants,)
    """
    # params 
    n_variants = len(v_trajs_data)
    if isinstance(trajs_data, dict):
        # expand ref traj to match dimension of v_trajs_data, but at the cost of more memory. 
        trajs_data = [trajs_data] * n_variants
    assert len(trajs_data) == len(v_trajs_data), "reference and variant traj data do not match shape."
    
    # computation case by case
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
            },
            "epsilon": config.epsilon,
        }
        # compute metric values
        metric_vals = np.zeros(n_variants)
        n_episodes = len(trajs_data[0]["obs"])
        for i in tqdm.tqdm(range(n_variants)):
            mval = 0.
            for j in range(n_episodes):
                mval += metric_func(trajs_data[i]["obs"][j], 
                                    v_trajs_data[i]["obs"][j], 
                                    **metric_kwargs)
            metric_vals[i] = mval / n_episodes
            print(colored("metric: {}".format(metric_vals[i]), "blue"))     
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
        encode_data_kwargs = {
            "tuple_length": config.tuple_length,
            "include_action": not config.not_include_action,
        }
        for i in tqdm.tqdm(range(n_variants)):
            processed_data = encode_data(trajs_data[i], **encode_data_kwargs)
            processed_v_data = encode_data(v_trajs_data[i], **encode_data_kwargs)
            metric_vals[i] = metric_func(processed_data, 
                                         processed_v_data, 
                                         **metric_kwargs)
            print(colored("metric: {}".format(metric_vals[i]), "blue"))     
    else:
        raise NotImplementedError("The given similarity metric is not available for metric computation.")
    return metric_vals



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
            },
            "epsilon": config.epsilon,
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
    
    
def test_metric_to_gt(config):
    """Compares ctrl-rollouted trajs to the ground truth reference traj.
    """
    # params
    n_episodes = config.n_episodes
    
    # different ref trajs
    variant_specs = [
        # NOTE: vanilla 
        # {"key": "trajectory_type", "vals": ["circle"]},
        # {"key": "trajectory_scale", "vals": [0.8]},
        # {"key": "num_cycles", "vals": [1, 2]},
        # NOTE: actual 
        {"key": "trajectory_type", "vals": ["circle", "square", "figure8"]},
        {"key": "trajectory_scale", "vals": [0.8, 1.0, 1.2]},
        {"key": "num_cycles", "vals": [1, 2, 3]},
        # NOTE: wide 
        # {"key": "trajectory_scale", "vals": [0.6, 0.8, 1.0, 1.2, 1.4]},
        # {"key": "trajectory_scale", "vals": [0.8, 1.0, 1.2]},
        # {"key": "num_cycles", "vals": [1, 2, 3, 4, 5]},
    ]
    traj_specs = [list(combo) for combo in itertools.product(*[d["vals"] for d in variant_specs])]
    n_variants = len(traj_specs)
    
    v_trajs_data,v_metrics = [], []
    ref_trajs = []
    for i, v_spec in enumerate(tqdm.tqdm(traj_specs)):
        v_config = get_variant_config({}, config)
        for j, val in enumerate(v_spec):
            setattr(v_config.task_config.task_info, variant_specs[j]["key"], val)
                
        exp = make_experiment(config, config.seed+i)
        trajs_data_, metrics_ = exp.run_evaluation(n_episodes=n_episodes, verbose=False)
        ref_traj = exp.env.X_GOAL   # (T,D)
        exp.close()
        
        # special processing (T,D) -> (T+1,D)
        ref_traj = np.concatenate([ref_traj, ref_traj[-1:]], 0)
        # remove goal info in obs for RL (T+1,D*k) -> (T+1,D) 
        if hasattr(config.task_config, "obs_goal_horizon") and config.task_config.obs_goal_horizon > 0:
            state_dim = trajs_data_["obs"][0][0].shape[0] // (config.task_config.obs_goal_horizon + 1)
            for j in range(len(trajs_data_["obs"])):    # each episode
                trajs_data_["obs"][j] = trajs_data_["obs"][j][:, :state_dim]
        
        v_trajs_data.append(trajs_data_)
        v_metrics.append(metrics_)
        # repeat ref traj to match dimensions
        # NOTE: there's no (meaningful) reference actions  
        ref_traj_dict = {"obs": [ref_traj]*n_episodes}
        ref_trajs.append(ref_traj_dict) 
        
    # compute metric values over tasks (different ref trajs)
    metric_vals = compute_metric(config, ref_trajs, v_trajs_data)
    
    # save results 
    os.makedirs(config.eval_output_dir, exist_ok=True)
    file_name = os.path.join(config.eval_output_dir, config.file_name)
    results = {
        "variant_specs": variant_specs,
        "traj_specs": traj_specs,
        "metric_vals": metric_vals.tolist(),
    }
    with open(file_name, "w")as f:
        yaml.dump(results, f, default_flow_style=False)
    print("test done ...")
    
    
def test_metric_to_learning(config):
    """
    """
    print("test done ...")
    

def test_metric_convergence(config):
    """"""
    # TODO: how to decompose to batch computation?
    # params 
    state_dim = 6
    
    # ref data 
    base_data_dir = getattr(config, "base_data_dir", None)
    base_data_paths = config.base_data_paths
    if base_data_paths is None:
        assert base_data_dir is not None, "Must provide base_data_dir to get all paths inside."
        base_data_paths = os.listdir(base_data_dir)
    ref_trajs = load_data_from_hdf5(base_data_paths, base_data_dir)
    # truncate obs to state
    for j in range(len(ref_trajs["obs"])):    # each episode
        ref_trajs["obs"][j] = ref_trajs["obs"][j][:, :state_dim]
    
    # variant data 
    variants = []
    v_trajs_data = []
    variant_data_dir = getattr(config, "variant_data_dir", None)
    for variant in config.variants:
        variants.append(variant.name)
        variant_data_paths = variant.data_paths
        if isinstance(variant_data_paths, str):
            variant_data_paths = [variant_data_paths]
        v_trajs = load_data_from_hdf5(variant_data_paths, variant_data_dir)
        # truncate obs to state
        for j in range(len(v_trajs["obs"])):    # each episode
            v_trajs["obs"][j] = v_trajs["obs"][j][:, :state_dim]
        v_trajs_data.append(v_trajs)
    
    # metric 
    metric_vals = compute_metric(config, ref_trajs, v_trajs_data)

    # save results 
    os.makedirs(config.eval_output_dir, exist_ok=True)
    file_name = os.path.join(config.eval_output_dir, config.file_name)
    results = {
        "variants": variants,
        "metric_vals": metric_vals.tolist(),
    }
    with open(file_name, "w")as f:
        yaml.dump(results, f, default_flow_style=False)

    # plot 
    if config.plot:
        plt.plot(variants, metric_vals)
        plt.xlabel(config.xlabel)
        plt.ylabel(config.metric_name)
        plt.title("Similarity Metric Convergence - {}".format(config.variant_name))
        fig_name = os.path.join(config.eval_output_dir, config.fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.show()
    print("test done ...")


# -----------------------------------------------------------------------------------
#                   Data Collection
# -----------------------------------------------------------------------------------

def convert_traj_infos_to_json_string(infos):
    """"""
    new_infos = [
        {k: v.tolist() if isinstance(v, np.ndarray) else v 
        for k, v in info.items()} 
        for info in infos
    ]
    return json.dumps(new_infos) 


def convert_json_string_to_traj_infos(json_infos):
    """"""
    return json.loads(json_infos)


def save_list_defaultdict_to_hdf5(file, data):
    """"""
    # new group for each field 
    for k, v in data.items():
        assert isinstance(v, list)
        group = file.create_group(k)
        # new dataset for each traj 
        for i in range(len(v)):
            if isinstance(v[0], (int, float, np.ndarray)) and k != "info":
                k_traj_data = np.asarray(v[i])
            else:
                # convert to str if can't be saved as numpy array
                # k_traj_data = pickle.dumps(v[i])
                try:
                    k_traj_data = convert_traj_infos_to_json_string(v[i])
                except:
                    break
            group.create_dataset(str(i), data=k_traj_data)


def load_data_from_hdf5(data_paths, data_dir=None):
    """Merges data from given HDF5 files."""
    # params 
    include_data_keys = ["obs", "action", "state"]
    
    # load data 
    data = defaultdict(list)
    for data_path in data_paths:
        if data_dir:
            data_path = os.path.join(data_dir, data_path)
        with h5py.File(data_path, "r") as f:
            n_episodes = f.attrs["n_episodes"]
            traj_data_group = f["trajectory_data"]
            for k in traj_data_group.keys():
                if k not in include_data_keys:
                    continue 
                for i in range(n_episodes):
                    if k == "info":
                        k_traj_data = convert_json_string_to_traj_infos(traj_data_group[k][i][()])
                    else:
                        k_traj_data = traj_data_group[k][str(i)][()]
                    data[k].append(k_traj_data)
    return data 


def collect_data_to_hdf5(config):
    """Run a ctrl on a task to collect trajectory data.
        Data is saved as in a HDF5 file. 
    """
    # params
    n_episodes = config.n_episodes
    
    # collect data 
    exp = make_experiment(config, config.seed)
    trajs_data_, metrics_ = exp.run_evaluation(n_episodes=n_episodes, verbose=False)

    # save data 
    hdf5_path = os.path.join(config.eval_output_dir, config.hdf5_file_name)
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        # TODO: should fix the weird format here (need to access 1st element)
        ctrl_data = trajs_data_.pop("controller_data", None)
        ctrl_data = ctrl_data[0] if ctrl_data else None
        sf_data = trajs_data_.pop("safety_filter_data", None)
        sf_data = sf_data[0] if sf_data else None
        # traj data  
        traj_group = f.create_group("trajectory_data")
        save_list_defaultdict_to_hdf5(traj_group, trajs_data_)
        # TODO: check, mpc has non-jsonable data 
        # if ctrl_data:
        #     ctrl_group = f.create_group("controller_data")
        #     save_list_defaultdict_to_hdf5(ctrl_group, ctrl_data)
        # if sf_data:
        #     sf_group = f.create_group("safety_filter_data")
        #     save_list_defaultdict_to_hdf5(sf_group, sf_data)
        # meta data 
        f.attrs["config"] = json.dumps(config)
        f.attrs["n_episodes"] = n_episodes
        f.attrs["algo"] = config.algo 
        f.attrs["task"] = config.task 
        if config.restore:
            f.attrs["restore"] = config.restore 
        if config.checkpoint:
            f.attrs["checkpoint"] = config.checkpoint 
        f.attrs["seed"] = config.seed 
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
            for k, v in result["correlation_metrics"].items():
                correlation_data[legend][k].append(v)
    
    # save to csv
    csv_mtx = np.zeros((len(legend_order), len(correlation_metric_order)))
    for i, legend in enumerate(legend_order):
        for j, k in enumerate(correlation_metric_order):
            cor_metric = np.asarray(correlation_data[legend][k])
            csv_mtx[i][j] = cor_metric.mean()
    header = ["Correlation"] + correlation_metric_order
    csv_rows = csv_mtx.tolist()
    csv_rows = [[l] + r for l, r in zip(legend_order, csv_rows)]
    csv_path = os.path.join(config.eval_output_dir, config.csv_file_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print("plot done ...")
    
    
def plot_metric_box(config):
    """Makes the box plot that compares each ctrl's distribution of simlarity score of its rollout trajs 
        to the ground truth ref traj among multiple tasks in each box. 
    """
    # # NOTE: for debug/testing
    # legend_map = {
    #     "pid": "PID",
    #     "pid1": "LQR",
    #     "pid2": "MPC",
    #     "pid3": "PPO untrained",
    #     "pid4": "PPO trained",
    # }
    legend_map = {
        # "random": "Random",
        "pid": "PID",
        "lqr": "LQR",
        "mpc": "MPC",
        "ppo_untrained": "PPO 100k",
        "ppo": "PPO 5m",
    }
    legend_order = [
        # "Random", 
        "PID", 
        "LQR", 
        "MPC", 
        ["PPO 100k", "PPO 5m"],
    ]
    
    # params
    dx = 1
    ddx = 0.75
    
    # load results 
    data = defaultdict(list)
    for d, legend in legend_map.items():
        algo_seed_dirs_pat = os.path.join(config.eval_output_dir, d, "seed*")        
        algo_seed_dirs = glob.glob(algo_seed_dirs_pat)

        for seed_dir in algo_seed_dirs:
            eval_file = os.path.join(seed_dir, config.file_name)
            print(eval_file)
            result = read_file(eval_file)
            metric_vals = result["metric_vals"]
            data[legend].append(metric_vals)
    for l in data:
        data[l] = np.asarray(data[l])   # (n_seeds, n_variants)
        
    # plot 
    fig = plt.figure(figsize=(10, 3))
    axes = fig.subplots(nrows=1, ncols=1)
    
    legend_order_flatten = sum([[l] if isinstance(l, str) else l for l in legend_order], [])
    positions = []
    x = 0
    for l in legend_order:
        if isinstance(l, str):
            x += dx
            positions.append(x)
        elif isinstance(l, list):
            for i, ll in enumerate(l):
                if i == 0:
                    x += dx
                else:
                    x += ddx 
                positions.append(x)
        else:
            raise TypeError("legend order elements must be str or list of str.")
    # mean over seeds
    y_means = [data[l].mean(0) for l in legend_order_flatten] 
    axes.boxplot(y_means, positions=positions, showfliers=False)
    
    axes.set_xticklabels(legend_order_flatten, rotation=-90)
    axes.set_ylabel("Metric score")
    axes.set_title("Distribution of {} to tasks among controllers".format(config.metric_name))
        
    fig_name = os.path.join(config.eval_output_dir, config.fig_name)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
    print("plot done ...")
    
    
def plot_metric_learning(config):
    """
    """
    
    
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
    fac.add_argument("--plot", action="store_true", help="if to plot the results.")
    fac.add_argument("--csv_file_name", type=str, default="temp.csv", help="csv file name to save.")
    fac.add_argument("--metric_name", type=str, default="LSED", help="metric name to appear on figure.")
    fac.add_argument("--hdf5_file_name", type=str, default="temp.hdf5", help="HDF5 file name to save.")
    fac.add_argument("--data_dir", type=str, help="common folder path to data files.")
    fac.add_argument("--data_paths", nargs='+', type=str, help="paths to data files.")

    fac.add_argument("--tuple_length", type=int, default=1)
    fac.add_argument("--not_include_action", action="store_true", help="if to remove actions for distribution metric when encoding data.")
    fac.add_argument("--mmd_mode", type=str, default="gaussian")
    fac.add_argument("--mmd_sigma", type=float, default=10.0)
    fac.add_argument("--geom_loss_func", type=str, default="sinkhorn")
    fac.add_argument("--geom_loss_p", type=int, default=2)
    fac.add_argument("--geom_loss_blur", type=float, default=0.05)
    fac.add_argument("--geom_loss_cost", type=str, help="distance/cost function between 2 samples to use for `sinkhorn` loss.")
    fac.add_argument("--geom_loss_kernel", type=str, help="same as argument `cost`, but for `hausdorff` loss.")
    fac.add_argument("--epsilon", type=float, default=0.05, help="distance threshold to determine if 2 states are the same.")

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
