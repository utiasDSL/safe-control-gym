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

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs, window_func, load_from_log_file, COLORS, LINE_STYLES
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video


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
    
    obs, info = env.reset()
    ep_obs, ep_act = [deepcopy(obs[:env.state_dim])], []

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
        ep_obs.append(deepcopy(obs[:env.state_dim]))
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
            
            ep_obs_list.append(np.stack(ep_obs))
            ep_act_list.append(np.stack(ep_act))
            ep_obs, ep_act = [deepcopy(obs[:env.state_dim])], []
            
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
    }
    if len(frames) > 0:
        eval_results["frames"] = frames
    # Other episodic stats from evaluation env.
    if len(env.queued_stats) > 0:
        queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
        eval_results.update(queued_stats)
    return eval_results


def save_traj_to_csv(config, env, results, save_dir):
    """ """
    ep_obs_list = results["ep_obs_list"]
    ep_act_list = results["ep_act_list"]
    header = ["t", "x", "x_dot", "z", "z_dot", "theta", "theta_dot", "action1", "action2"]
    dim = 9
    state_dim = 6
    action_dim = 2
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
        thrust = np.clip(actions, env.action_space.low, env.action_space.high)
        if config.task_config.normalized_rl_action_space:
            thrust = (1 + env.norm_act_scale * thrust) * env.hover_thrust
        data[:-1, -action_dim:] = thrust  
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
    n_episodes = len(ep_obs_list)
    # process trajectories
    # (T, state_dim)
    traj_goal = env.X_GOAL
    if env.QUAD_TYPE == 1:
        state_names = ["z", "z_dot"]
        state_dim = 2
        act_dim = 1
    elif env.QUAD_TYPE == 2:
        state_names = ["x", "x_dot", "z", "z_dot", "theta", "theta_dot"]
        state_dim = 6
        act_dim = 2
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
        
        for j in range(state_dim):
            traj_goal_dim = traj_goal[:, j]
            traj_dim = traj[:, j]
            axe[j].plot(x[1:], traj_goal_dim, color="green")
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


def plot_phase_traj(env, results, output_path="temp.png", max_episode_plot=1):
    """Plot the trajectory in phase (state/velocity) spaces."""
    ep_obs_list = results["ep_obs_list"]
    ep_act_list = results["ep_act_list"]
    n_episodes = len(ep_obs_list)
    marker_size = 100
    # process trajectories
    # (T, state_dim)
    traj_goal = env.X_GOAL
    if env.QUAD_TYPE == 1:
        state_names = ["z", "z_dot"]
        state_dim = 2
        ncols = 2
    elif env.QUAD_TYPE == 2:
        state_names = ["x", "x_dot", "z", "z_dot", "theta", "theta_dot"]
        state_dim = 6
        ncols = 3
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
        traj = ep_obs_list[i]
        
        # (T+1 or T, 2)
        if env.QUAD_TYPE == 1:
            z_goal, z_dot_doal = traj_goal[:, 0], traj_goal[:, 1]
            z, z_dot = traj[:, 0], traj[:, 1]
            x = np.zeros_like(z)
            
            axe[0].plot(x[1:], z_goal, color="green")
            # axe[0].scatter(x[-1], z_goal[-1], color="green", s=marker_size)
            axe[0].plot(x, z, color="black")
            axe[0].scatter(x[-1], z[-1], color="red", s=marker_size//2)
            axe[0].set_xlabel("x")
            axe[0].set_ylabel("z")
            axe[0].set_title("z traj")

            axe[1].plot(x[1:], z_dot_doal, color="green")
            # axe[1].scatter(x[-1], z_dot_doal[-1], color="green", s=marker_size)
            axe[1].plot(x, z_dot, color="black")
            axe[1].scatter(x[-1], z_dot[-1], color="red", s=marker_size//2)
            axe[1].set_xlabel("x_dot")
            axe[1].set_ylabel("z_dot")
            axe[1].set_title("z_dot traj")
            
        elif env.QUAD_TYPE == 2:
            x_goal, x_dot_goal, z_goal, z_dot_doal = traj_goal[:, 0], traj_goal[:, 1], traj_goal[:, 2], traj_goal[:, 3]
            x, x_dot, z, z_dot = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
            theta, theta_dot = traj[:, 4], traj[:, 5]

            axe[0].plot(x_goal, z_goal, color="green")
            # axe[0].scatter(x_goal[-1], z_goal[-1], color="green", s=marker_size)
            axe[0].plot(x, z, color="black")
            axe[0].scatter(x[-1], z[-1], color="red", s=marker_size//2)
            axe[0].set_xlabel("x")
            axe[0].set_ylabel("z")
            axe[0].set_title("x-z traj")
            
            axe[1].plot(x_dot_goal, z_dot_doal, color="green")
            # axe[1].scatter(x_dot_goal[-1], z_dot_doal[-1], color="green", s=marker_size)
            axe[1].plot(x_dot, z_dot, color="black")
            axe[1].scatter(x_dot[-1], z_dot[-1], color="red", s=marker_size//2)
            axe[1].set_xlabel("x_dot")
            axe[1].set_ylabel("z_dot")
            axe[1].set_title("x_dot-z_dot traj")
            
            axe[2].plot(theta, theta_dot, color="black")
            axe[2].scatter(theta[-1], theta_dot[-1], color="red", s=marker_size//2)
            axe[2].set_xlabel("theta")
            axe[2].set_ylabel("theta_dot")
            axe[2].set_title("theta-theta_dot traj")
    # save fig 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print("Test trajectories (phase) plotted.")
    
    
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
