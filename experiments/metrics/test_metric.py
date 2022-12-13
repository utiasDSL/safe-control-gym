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
from functools import partial
import matplotlib.pyplot as plt
from termcolor import colored
import tqdm 

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import merge_dict
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.math_and_models.metrics.similarity_metrics import *


# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------






# -----------------------------------------------------------------------------------
#                   Test
# -----------------------------------------------------------------------------------
    

def test_trajectory_metric(config):
    """"""
    # params 
    n_episodes = config.n_episodes
    n_variants = len(config.variant_config)
    env_seed_offset = 100

    # base/control group.
    env_func = partial(make, config.task, **config.task_config)
    env = env_func()
    env.seed(config.seed)
    ctrl = make(config.algo, env_func, seed=config.seed)
    experiment = BaseExperiment(env, ctrl)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, verbose=False)
    experiment.close()
    
    # variants/experimental group.
    v_trajs_data = []
    v_metrics = []
    for i, v_config in enumerate(config.variant_config):
        v_task_config = deepcopy(config.task_config) 
        merge_dict(v_task_config, v_config)
        
        env_func = partial(make, config.task, **v_task_config)
        env = env_func()
        if config.metric in ["lsed", "dtw", "edr", "lcss", "discrete_frechet"]:
            # requires the same initial conditions btw reference traj and variant traj
            env.seed(config.seed)           
        elif config.metric in ["mmd_loss"]:
            # only require random samples from reference traj and variant traj
            # no need for matching initial conditions, here we use a different seed for each variant
            env.seed(config.seed + env_seed_offset + i)
        else:
            raise NotImplementedError("The given similarity metric is not available for data collection.")
        ctrl = make(config.algo, env_func, seed=config.seed)
        experiment = BaseExperiment(env, ctrl)
        trajs_data_, metrics_ = experiment.run_evaluation(n_episodes=n_episodes, verbose=False)
        experiment.close()
        
        v_trajs_data.append(trajs_data_)
        v_metrics.append(metrics_)
        
    # metrics 
    if config.metric in ["lsed", "dtw", "edr", "lcss", "discrete_frechet"]: 
        # compute metric of each pair of ref traj and variant traj, use the average as final metric value 
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
        metric_vals = np.zeros(n_variants)
        for i in tqdm.tqdm(range(n_variants)):
            mval = 0.
            for j in range(n_episodes):
                mval += metric_func(trajs_data["obs"][j], 
                                    v_trajs_data[i]["obs"][j], 
                                    **metric_kwargs)
            metric_vals[i] = mval / n_episodes
            print(colored("param: {} | metric: {}".format(config.variants[i], metric_vals[i]), "blue"))     
    elif config.metric in ["mmd_loss"]:
        # compute metric btw batch of ref traj samples and variant traj samples
        metric_func = eval(config.metric)
        metric_kwargs = {
            "mode": config.mmd_mode,
            "sigma": config.mmd_sigma,
        }
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
        
    # save to file 
    os.makedirs(config.eval_output_dir, exist_ok=True)
    file_name = os.path.join(config.eval_output_dir, config.file_name)
    results = {
        "reference": config.reference,
        "variants": config.variants,
        "xlabel": config.xlabel,
        "metric_vals": metric_vals.tolist(),
        "performance_metrics": v_metrics,
    }
    with open(file_name, "w")as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # plot 
    if config.plot:
        fig = plt.figure(figsize=(8, 3))
        axes = fig.subplots(nrows=1, ncols=2)
        # metric plot 
        axes[0].plot(config.variants, metric_vals) #, alpha=0.7)
        # plt.fill_between(params, y_quant_3rd, y_quant_1st, alpha=0.3)
        axes[0].axvline(x=float(config.reference), linewidth=1, color='r', linestyle="--", label="reference value")
        axes[0].set_xlabel(config.xlabel)
        axes[0].set_ylabel("{} distance to reference".format(config.metric))
        axes[0].set_title("{}".format(config.metric))
        # rmse plot
        rmse_vals = [vm["average_rmse"] for vm in v_metrics]
        axes[1].plot(config.variants, rmse_vals) #, alpha=0.7)
        axes[1].axvline(x=float(config.reference), linewidth=1, color='r', linestyle="--", label="reference value")
        axes[1].set_xlabel(config.xlabel)
        axes[1].set_ylabel("average RMSE")
        axes[1].set_title("RMSE")
        
        fig_name = os.path.join(config.eval_output_dir, config.fig_name)
        plt.savefig(fig_name)
        plt.show()
    
    print("test done ...")    
    



    
    
    
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
    fac.add_argument("--n_episodes", type=int, default=10, help="number of test episodes.")
    fac.add_argument("--metric", type=str, default="mse", help="metric name.")
    fac.add_argument("--fig_name", type=str, default="temp.png", help="fig name to save.")
    fac.add_argument("--file_name", type=str, default="temp.yaml", help="file name to save.")
    fac.add_argument("--tuple_length", type=int, default=1)
    fac.add_argument("--mmd_mode", type=str, default="gaussian")
    fac.add_argument("--mmd_sigma", type=float, default=10.0)
    fac.add_argument("--plot", action="store_true", help="if to plot the results.")


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
