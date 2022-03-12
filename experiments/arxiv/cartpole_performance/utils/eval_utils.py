import os
import sys
from functools import partial
import os
import pickle
import sys
import torch
from copy import deepcopy
import numpy as np 
import matplotlib.pyplot as plt 
import csv
from collections import defaultdict
import math 
import glob
import re 
from dict_deep import deep_set
import pandas as pd 

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs, window_func, load_from_log_file, COLORS, LINE_STYLES
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics


##############################################################################################
#############################   Testing Utils   ##############################################
##############################################################################################

def make_fixed_init_env(config, env_seed=None, n_episodes=10, init_state=None):
    """"""
    assert init_state is not None, "Must provide an init_state dict for the env."
    config = deepcopy(config_origin)
    # modify env config with fixed init 
    config.task_config.randomized_init = False 
    config.task_config.init_state_randomization_info = None 
    # use deep_set since init_state is None, cannot assign 
    deep_set(config, "task_config.init_state", {})
    for state_k, state_v in init_state.items():
        deep_set(config, "task_config.init_state.{}".format(state_k), state_v)
    # make env 
    env_func = partial(make, config.task, seed=env_seed, output_dir=config.output_dir, **config.task_config)
    env = env_func()
    env = RecordEpisodeStatistics(env, n_episodes)
    env.add_tracker("constraint_violation", 0, mode="queue")
    env.add_tracker("constraint_values", 0, mode="queue")
    env.add_tracker("mse", 0, mode="queue")
    return env 


def run_with_env(config, ctrl, env, render=False, n_episodes=10, verbose=False):
    """Test policy with a given environment and save test trajectories."""
    ep_obs_list, ep_act_list = [], []
    ep_returns, ep_lengths = [], []
    frames = []
    goals = []
    
    obs, info = env.reset()
    ep_obs, ep_act = [deepcopy(obs)], []
    goals.append(deepcopy(env.X_GOAL))

    obs = ctrl.obs_normalizer(obs)
    c = info.get("constraint_values", 0)
    
    while len(ep_returns) < n_episodes:
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(ctrl.device)
            c = torch.FloatTensor(c).to(ctrl.device)
            
            if config.algo == "sac" or config.algo == "ddpg":
                action = ctrl.agent.ac.act(obs, deterministic=True)
            elif config.algo == "ppo":
                action = ctrl.agent.ac.act(obs)
            elif config.algo == "safe_explorer_ppo":
                action = ctrl.agent.ac.act(obs, c=c)
            else:
                raise NotImplementedError("ac.act not implemented.")
            
        obs, reward, done, info = env.step(action)
        ep_obs.append(deepcopy(obs))
        ep_act.append(deepcopy(action))

        if render:
            env.render()
            frames.append(env.render("rgb_array"))
        if verbose:
            print("obs {} | act {}".format(obs, action))

        if done:
            assert "episode" in info
            ep_returns.append(info["episode"]["r"])
            ep_lengths.append(info["episode"]["l"])
            obs, _ = env.reset()
            goals.append(deepcopy(env.X_GOAL))
            
            ep_obs_list.append(np.stack(ep_obs))
            ep_act_list.append(np.stack(ep_act))
            ep_obs, ep_act = [deepcopy(obs)], []
            
        obs = ctrl.obs_normalizer(obs)
        c = info.get("constraint_values", 0)
     
    # collect evaluation results
    ep_lengths = np.asarray(ep_lengths)
    ep_returns = np.asarray(ep_returns)
    eval_results = {
        "ep_returns": ep_returns, 
        "ep_lengths": ep_lengths, 
        "ep_obs_list": ep_obs_list,
        "ep_act_list": ep_act_list,
        "goals": goals,
    }
    if len(frames) > 0:
        eval_results["frames"] = frames
    # Other episodic stats from evaluation env.
    if len(env.queued_stats) > 0:
        queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
        eval_results.update(queued_stats)
    return eval_results


def save_traj_to_csv(config, env, results, save_dir):
    """Saves the test trajectories to csv files."""
    ep_obs_list = results["ep_obs_list"]
    ep_act_list = results["ep_act_list"]
    header = ["t", "x", "x_dot", "theta", "theta_dot", "action"]
    dim = 6
    state_dim = 4
    action_dim = 1    
    n_episodes = len(ep_obs_list)
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(n_episodes):
        # (T+1, state_dim)
        states = ep_obs_list[i]
        # (T, action_dim)
        actions = ep_act_list[i]
        Tp1 = len(states)
        x = np.arange(Tp1)
        
        data = np.zeros((Tp1, dim))
        data[:, 0] = x 
        data[:, 1:1+state_dim] = states
        force = np.clip(actions, env.action_space.low, env.action_space.high)
        if config.task_config.normalized_rl_action_space:
            force = env.action_scale * force
        data[:-1, -action_dim:] = force 
        
        rows = data.tolist() 
        csv_path = os.path.join(save_dir, "traj_{}.csv".format(i))
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)    
    print("Test trajectories (time) saved to csv files.")


def plot_time_traj(env, results, output_path="temp.png", max_episode_plot=1):
    """Plot each dimension of state and action in time."""
    ep_obs_list = results["ep_obs_list"]
    ep_act_list = results["ep_act_list"]
    goals = results["goals"]
    n_episodes = len(ep_obs_list)
    
    # process trajectories
    state_names = ["x", "x_dot", "theta", "theta_dot"]
    state_dim = 4 
    act_dim = 1
    ncols = state_dim + 1
    n_rows = min(n_episodes, max_episode_plot)
    
    # fig = plt.figure()
    h_size = ncols * 4
    v_size = n_rows * 3
    fig = plt.figure(figsize=(h_size, v_size))
    axes = fig.subplots(nrows=n_rows, ncols=ncols)
       
    for i in range(n_rows):
        if n_rows == 1:
            axe = axes
        else:
            axe = axes[i]
        # (T+1, state_dim), (T, act_dim)
        traj, acts = ep_obs_list[i], ep_act_list[i]
        x = list(range(len(traj)))
        goal = goals[i]
        traj_goal = np.stack([goal]*len(traj))
        
        for j in range(state_dim):
            traj_goal_dim = traj_goal[:, j]
            traj_dim = traj[:, j]
            axe[j].plot(x, traj_goal_dim, color="green")
            axe[j].plot(x, traj_dim, color="black")
            if i == 0:
                axe[j].set_title(state_names[j])
            
        for k in range(act_dim):
            action_dim = acts[:, k]
            axe[-1].plot(x[:-1], action_dim)
            if i == 0:
                axe[-1].set_title("action")
    # save fig 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print("Test trajectories (time) plotted.")
    
    
def test_robustness_with_fixed_seeds(config, 
                                     ctrl, 
                                     render=False, 
                                     n_episodes=10, 
                                     verbose=False, 
                                     **kwargs):
    """Test policy for robustness tests, with fixed seeds (to match other control baselines)."""
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    
    def make_env(seed):
        env = env_func(seed=seed)
        env = RecordEpisodeStatistics(env, n_episodes)
        env.add_tracker("constraint_violation", 0, mode="queue")
        env.add_tracker("constraint_values", 0, mode="queue")
        env.add_tracker("mse", 0, mode="queue")
        return env    
    env_queued_stats = defaultdict(list)
    
    ep_obs_list, ep_act_list = [], []
    ep_returns, ep_lengths = [], []
    frames = []
    goals = []
    
    seed = 1 
    env = make_env(seed)
    obs, info = env.reset()
    # print("seed: {}, obs: {}".format(seed, obs))
    
    ep_obs, ep_act = [deepcopy(obs)], []
    goals.append(deepcopy(env.X_GOAL))

    obs = ctrl.obs_normalizer(obs)
    c = info.get("constraint_values", 0)
    
    while len(ep_returns) < n_episodes:
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(ctrl.device)
            c = torch.FloatTensor(c).to(ctrl.device)
            
            if config.algo == "sac" or config.algo == "ddpg":
                action = ctrl.agent.ac.act(obs, deterministic=True)
            elif config.algo == "ppo" or config.algo == "rarl" or config.algo == "rap":
                action = ctrl.agent.ac.act(obs)
            elif config.algo == "safe_explorer_ppo":
                action = ctrl.agent.ac.act(obs, c=c)
            else:
                raise NotImplementedError("ac.act not implemented.")
            
        obs, reward, done, info = env.step(action)
        ep_obs.append(deepcopy(obs))
        ep_act.append(deepcopy(action))

        if render:
            env.render()
            frames.append(env.render("rgb_array"))
        if verbose:
            print("obs {} | act {}".format(obs, action))

        if done:
            assert "episode" in info
            ep_returns.append(info["episode"]["r"])
            ep_lengths.append(info["episode"]["l"])
            
            if len(env.queued_stats) > 0:
                for k, v in env.queued_stats.items():
                    env_queued_stats[k].extend(v)
            
            # create new env 
            seed += 1
            env = make_env(seed)
            obs, _ = env.reset()
            # print("seed: {}, obs: {}".format(seed, obs))
            
            goals.append(deepcopy(env.X_GOAL))
            ep_obs_list.append(np.stack(ep_obs))
            ep_act_list.append(np.stack(ep_act))
            ep_obs, ep_act = [deepcopy(obs)], []
            
        obs = ctrl.obs_normalizer(obs)
        c = info.get("constraint_values", 0)
     
    # collect evaluation results
    ep_lengths = np.asarray(ep_lengths)
    ep_returns = np.asarray(ep_returns)
    eval_results = {
        "ep_returns": ep_returns, 
        "ep_lengths": ep_lengths, 
        "ep_obs_list": ep_obs_list,
        "ep_act_list": ep_act_list,
        "goals": goals,
    }
    if len(frames) > 0:
        eval_results["frames"] = frames
    # Other episodic stats from evaluation env.
    if len(env_queued_stats) > 0:
        queued_stats = {k: np.asarray(v) for k, v in env_queued_stats.items()}
        eval_results.update(queued_stats)
    return eval_results


##############################################################################################
#############################   Plotting Utils   #############################################
##############################################################################################

def reward_to_cost(y):
    """Converts RL reward to control cost (for plotting), 250 is total reward/return upper bound."""
    return 250 - y


def load_stats(legend_dir_specs, 
               scalar_names=[], 
               window=None, 
               x_num_max=None, 
               x_rescale_factor=None):
    """Gets all processed statistics for multiple scalars."""
    scalar_stats = {}
    for scalar_name in scalar_names:
        # Get all stats.
        stats = defaultdict(list)
        for l, dirs in legend_dir_specs.items():
            for d in dirs:
                # Pick from either log source (tensorboard or log text files).
                path = os.path.join(d, "logs", scalar_name + ".log")
                _, x, _, y = load_from_log_file(path)
                x, y = np.asarray(x), np.asarray(y)
                # quick hack to convert reward to cost (consistent with other control methods)
                if "return" in scalar_name.lower() or "reward" in scalar_name.lower():
                    y = reward_to_cost(y)
                # rescale the x-axis, e.g. convert to training step to time 
                if x_rescale_factor is not None:
                    x *= x_rescale_factor
                # Smoothing.
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
            # Record stats.
            processed_stats[name] = {
                "x": x,
                "mean": np.mean(y, axis=0),
                "std": np.std(y, axis=0),
                "median": np.median(y, axis=0),
                "quantiles": np.quantile(y, [0.25, 0.75], axis=0),
                "min": np.min(y, axis=0),
                "max": np.max(y, axis=0),
            }
        # Copy over stats.
        scalar_stats[scalar_name] = deepcopy(processed_stats)
    return scalar_stats
    
    
def plot_from_exps(legend_dir_specs,
                    out_path="temp.jpg",
                    scalar_names=[],
                    title="Traing Curves",
                    sub_titles=["Loss Curve"],
                    xlabel="Epochs",
                    ylabels=["Loss"],
                    window=None,
                    x_num_max=None,
                    x_rescale_factor=None,
                    num_std=1,
                    use_median_quantile=False,
                    cols_per_row=3):
    """Plots 1 statistic figure at a time."""
    # Get all stats.
    scalar_stats = load_stats(legend_dir_specs, 
                              scalar_names=scalar_names, 
                              window=window, 
                              x_num_max=x_num_max,
                              x_rescale_factor=x_rescale_factor)
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
        elif num_cols > 1:
            ax = axes[col_idx]
        else:
            ax = axes
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
        # ax.set_ylim((-10, None))
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
    fig.legend(lines,
               labels,
               loc='lower center',
               fancybox=True,
               shadow=True,
               borderaxespad=0.1,
               ncol=7)
    plt.savefig(out_path)
    plt.show()
    return scalar_stats


def save_stats_to_csv(scalar_stats,
                      algo_name_map=None, 
                      scalar_name_map=None,
                      csv_path="stats.csv"):
    """Saves the queried experiment statistics to a csv file."""
    curves = ["mean", "std", "min", "max", "median", "top_quartile", "bottom_quartile"]
    header = []
    stat_rows = []
    x_already = False
    for scalar_name in scalar_stats:
        stats = scalar_stats[scalar_name]
        true_scalar_name = scalar_name_map[scalar_name] if scalar_name_map else scalar_name
        # Collect stats.
        for algo_name in sorted(stats.keys()):
            true_algo_name = algo_name_map[algo_name] if algo_name_map else algo_name
            stat = stats[algo_name]
            # X.
            if not x_already:
                header.append("x-Step")
                stat_rows.append(stat["x"])
                x_already = True
            # Y.
            header.extend(["y-{}-{}-{}".format(true_scalar_name, true_algo_name, c) for c in curves])
            stat_rows.append(stat["mean"])
            stat_rows.append(stat["std"])
            stat_rows.append(stat["min"])
            stat_rows.append(stat["max"])
            stat_rows.append(stat["median"])
            stat_rows.append(stat["quantiles"][1])
            stat_rows.append(stat["quantiles"][0])
    # Make rows.
    stat_mtx = np.array(stat_rows).transpose()
    rows = stat_mtx.tolist()
    # Write to csv.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_eval_stats(legend_map, 
                    eval_result_dir=None, 
                    scalar_name=None, 
                    scalar_normalize_by_ep_length=True,
                    scalar_postprocess_func=None,
                    get_processed_stats=False):
    """Gets statistics for 1 scalar metric from a standalone evaluation folder."""
    assert (eval_result_dir is not None) and (scalar_name is not None)
    data = {}
    for d, legend in legend_map.items():  
        # get all seed dirs of the algo      
        full_dir_pattern = os.path.join(eval_result_dir, d, "seed*")        
        full_seed_dirs = glob.glob(full_dir_pattern)
        algo_data = []

        for seed_d in full_seed_dirs:
            # eval results files for different params 
            eval_result_paths = [f for f in os.listdir(seed_d)]
            seed_data = []
            
            for f_name in eval_result_paths:
                with open(os.path.join(seed_d, f_name), "rb") as f:
                    result = pickle.load(f)
                param_value = float(f_name.replace(".pkl", ""))
                # shape (n_episodes,)
                lengths = result["ep_lengths"]
                scalars = result[scalar_name]
                assert len(lengths) == len(scalars)
                # further process the scalar to our desired metric
                if scalar_normalize_by_ep_length:
                    scalars /= lengths
                if scalar_postprocess_func is not None:
                    scalars = scalar_postprocess_func(scalars)
                seed_data.append([param_value, scalars])

            # shape (#params, 2), where the 2 objs are a float and an array 
            seed_data = sorted(seed_data, key=lambda x: x[0])
            # shape (#params,) & (#params, n_episodes)
            param_values, param_episode_ys = list(zip(*seed_data))
            algo_data.append([param_values, param_episode_ys])
        
        # shape (#params,)
        param_values = np.asarray(algo_data[0][0])
        # shape (#seeds, #params, n_episodes)
        y = np.asarray([param_episode_ys for _, param_episode_ys in algo_data])
        data[legend] = [param_values, y]
        
    # post-process stats (across seeds and episodes)
    if get_processed_stats:
       data_processed = {
           legend: {
                "x": param_values,
                "mean": np.mean(y.mean(-1), axis=0),
                "std": np.std(y.mean(-1), axis=0),
                "median": np.median(y.mean(-1), axis=0),
                "quantiles": np.quantile(y.mean(-1), [0.25, 0.75], axis=0),
                "min": np.min(y.mean(-1), axis=0),
                "max": np.max(y.mean(-1), axis=0),
           }
           for legend, (param_values, y) in data.items()
       } 
       return data, data_processed
    return data 


def save_stats_to_excel(data,
                        csv_dir=None):
    """Saves a scalar data to one csv per algo."""
    for algo_name in sorted(data.keys()):
        # (#params,), (#seeds, #params, n_episodes)
        param_values, y = data[algo_name]
        header = param_values 
        # (n_episodes, #params)
        stat_mtx = np.mean(y, axis=0).transpose()
        rows = stat_mtx.tolist()
        
        # Write to csv.
        csv_path = os.path.join(csv_dir, algo_name, "data.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        # Write to excel for easier copy-paste.
        read_file = pd.read_csv (csv_path)
        excel_path = os.path.join(csv_dir, algo_name, "data.xlsx")
        read_file.to_excel (excel_path, index=None, header=True)
