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

from eval_utils import * 


##############################################################################################
#############################   Constants   ##################################################
##############################################################################################

# cartpole controller time interval 
dt = 1.0 / 50.0


##############################################################################################
#############################   Testing Functions   ##########################################
##############################################################################################

def test_policy(config):
    """Run the (trained) policy/controller for evaluation.
    
    Saves test trajectories to csv files.
    Supports evaluation with fixed initial conditions.
    
    """
    assert config.restore is not None, "must specify --restore for policy evaluation."
    assert config.task_config.done_on_out_of_bound is False, "post-evaluation should disable early termination."
    # Evaluation setup.
    set_device_from_config(config)
    if config.set_test_seed:
        # seed the evaluation (both controller and env) if given
        set_seed_from_config(config)
        env_seed = config.seed
    else:
        env_seed = None
    # Define function to create task/env.
    env_func = partial(make, config.task, seed=env_seed, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, config.restore_model_file))

    # Test controller.
    if config.fix_env_init:
        n_episodes = 1
        init_state = {
            "init_x": -0.06839607,
            "init_theta": -0.07790506,
            # "init_x_dot": -0.06486605,
            # "init_theta_dot": 0.03383949,
        }
        env = make_fixed_init_env(config, init_state=init_state, 
                                  env_seed=env_seed, n_episodes=n_episodes)
    else:
        n_episodes = config.algo_config.eval_batch_size
        env = control_agent.env 
    results = run_with_env(config, control_agent, env, n_episodes=n_episodes, 
                           render=config.render, verbose=config.verbose)
    
    # Save evalution results.
    if hasattr(config, "eval_output_dir") and config.eval_output_dir is not None:
        eval_output_dir = config.eval_output_dir
    else:
        eval_output_dir = os.path.join(config.output_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    # test trajs and statistics 
    eval_path = os.path.join(eval_output_dir, config.eval_output_path)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "wb") as f:
        pickle.dump(results, f)
    # time trajs for better visualization & plotting
    save_traj_to_csv(config, env, results, os.path.join(eval_output_dir, "eval_trajs"))
    plot_time_traj(env, results, 
                   output_path=os.path.join(eval_output_dir, "time_trajectory.png"), 
                   max_episode_plot=config.max_episode_plot)
    control_agent.close()
    print("Evaluation done.")


def test_from_checkpoints(config):
    """Use saved checkpoints to test policy throughout training.
    
    This function is used when evaluation uses different settings to that during training.
    Hence, we use saved checkpoints to run test episodes with the modified settings,
    e.g. here we calculate the normalized metrics, including the RMSE as a common metric
    to compare fairly with other control baselines.
    
    """
    assert config.restore is not None, "must specify --restore for policy evaluation."
    assert config.task_config.done_on_out_of_bound is False, "post-evaluation should disable early termination."
    # Get all checkpoints
    checkpoint_dir_full = os.path.join(config.restore, config.checkpoint_dir)
    assert os.path.exists(checkpoint_dir_full), "checkpoint_dir does not exist in {}".format(config.restore)
    checkpoint_re = re.compile("model_(.*)\.pt")
    checkpoints = [
        [float(checkpoint_re.search(pt).group(1)), pt] 
        for pt in os.listdir(checkpoint_dir_full)
    ]
    checkpoints = sorted(checkpoints, key=lambda x: x[0])

    # Evaluation setup.
    set_device_from_config(config)
    if config.set_test_seed and config.set_test_seed_as_training_eval:
        raise ValueError("Can only enable either `set_test_seed` or `set_test_seed_as_training_eval`.")
    elif config.set_test_seed:
        set_seed_from_config(config)
        env_seed = config.seed
    elif config.set_test_seed_as_training_eval:
        # algo seeding uses seed for original training 
        set_seed_from_config(config)
        # env seeding uses seed for eval_env back in during training 
        env_seed = config.seed * 111
    else:
        env_seed = None 
    
    # Define function to create task/env.
    env_func = partial(make, config.task, seed=env_seed, output_dir=config.output_dir, **config.task_config)
    # create env and send to .run() explicitly
    # since if not, the default self.env has infinit buffer but not of size eval_batch_size
    env = env_func()
    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    
    # Test on each checkpoint 
    test_results = defaultdict(list)
    for i, (step, checkpoint) in enumerate(checkpoints):
        print("Progress {}, step {} evaluating...".format(i / len(checkpoints), step))
        checkpoint_path_full = os.path.join(checkpoint_dir_full, checkpoint)
        control_agent.load(checkpoint_path_full)
        results = control_agent.run(env=env, n_episodes=config.algo_config.eval_batch_size)
        
        test_results["step"].append(step)
        ep_lengths = results["ep_lengths"]
        for k, v in results.items():
            if not isinstance(v, np.ndarray):
                pass 
            # raw mean over n_episodes 
            test_results[k].append(v.mean())
            # normalized mean 
            if k is not "ep_lengths":
                assert len(v) == len(ep_lengths)
                # don't use `v / ep_lengths`, in case v is more than 1 dim
                normalized_v = np.asarray([v[i] / ep_lengths[i] for i in range(len(v))])
                if "mse" in k:
                    # convert mse cost to rmse (as we desired for common metric across baselines)
                    stat = np.mean(np.sqrt(normalized_v))
                    name = k.replace("mse", "rmse")
                else:
                    # take simple mean for other metrics such as constrasint violations 
                    stat = np.mean(normalized_v)
                    name = k
                test_results["normalized_" + name].append(stat)

    # Save evalution results.
    checkpoint_eval_dir_full = os.path.join(config.restore, "logs", config.checkpoint_eval_dir)
    os.makedirs(checkpoint_eval_dir_full, exist_ok=True)
    steps = test_results.pop("step")

    for k, v in test_results.items():
        scalar_name = "{}/{}".format(config.checkpoint_eval_dir, k)
        header = ["step", scalar_name]
        stat_mtx = np.array([steps, v]).transpose()
        rows = stat_mtx.tolist()
        
        csv_path = os.path.join(checkpoint_eval_dir_full, "{}.log".format(k))
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    control_agent.close()
    print("Checkpoint evaluation done.")


def test_policy_robustness(config):
    """Run the (trained) policy/controller for robustness evaluation.    
    """
    assert config.restore is not None, "must specify --restore for policy evaluation."
    assert config.task_config.done_on_out_of_bound is False, "post-evaluation should disable early termination."
    # Evaluation setup.
    set_device_from_config(config)
    if config.set_test_seed:
        # seed the evaluation (both controller and env) if given
        set_seed_from_config(config)
        env_seed = config.seed
    else:
        env_seed = None
    # Define function to create task/env.
    env_func = partial(make, config.task, seed=env_seed, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, config.restore_model_file))

    # Test controller under robustness evaluation settings.
    results = test_robustness_with_fixed_seeds(config, 
                                               control_agent, 
                                               render=False,
                                               n_episodes=config.algo_config.eval_batch_size)
    # Save evalution results.    
    if config.eval_output_dir is not None:
        eval_output_dir = config.eval_output_dir
    else:
        eval_output_dir = os.path.join(config.output_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    # test trajs and statistics 
    eval_path = os.path.join(eval_output_dir, config.eval_output_path)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "wb") as f:
        pickle.dump(results, f)
    control_agent.close()
    print("Robustness evaluation done.")


##############################################################################################
#############################   Performance   ################################################
##############################################################################################

def plot_performance(config):
    """Gets the plot and csv for performance (in RMSE)."""
    legend_map = {
        "ppo": "ppo",
        "sac": "sac",
    }
    algo_name_map = {
        "ppo": "PPO",
        "sac": "SAC",
    }
    scalar_name_map = {
        "checkpoint_eval/normalized_rmse": "Cost",
    }
    
    # Collect results.
    spec = {}
    for d, legend in legend_map.items():
        seed_dirs = os.listdir(os.path.join(config.plot_dir, d))
        spec[legend] = [os.path.join(config.plot_dir, d, sd) for sd in seed_dirs]

    # Collective plot (mean only).
    scalar_stats = plot_from_exps(
        spec, 
        out_path=os.path.join(config.plot_dir, "performance.jpg"), 
        scalar_names=["checkpoint_eval/normalized_rmse"], 
        title="Learning Curves", 
        sub_titles=["Average RMSE"], 
        xlabel="Training Time (s)", 
        ylabels=["Cost"], 
        window=None, 
        x_num_max=None, 
        x_rescale_factor=dt,
        num_std=0, 
        use_median_quantile=True, 
        cols_per_row=3
    )
    
    # Save stats to CSV.
    save_stats_to_csv(
        scalar_stats, 
        algo_name_map=algo_name_map,  
        scalar_name_map=scalar_name_map, 
        csv_path=os.path.join(config.plot_dir, "performance.csv")
    )
    print("Performance plotting done.")
    
    
##############################################################################################
#############################   Constraints   ################################################
##############################################################################################
  
def plot_constraint(config):
    """Gets the plot and csv for total constraint violations (throughout training)."""
    legend_map = {
        "ppo": "ppo",
        # "safe_explorer_ppo/slack0.050.03": "seppo_0.05-0.03",
        "safe_explorer_ppo/slack0.050.07": "seppo_0.05-0.07",
    }
    algo_name_map = {
        "ppo": "PPO",
        # "seppo_0.05-0.03": "PPO safety_layer theta_slack0.05 theta_dot_slack0.03",
        "seppo_0.05-0.07": "PPO safety_layer theta_slack0.05 theta_dot_slack0.07",
    }
    scalar_name_map = {
        "checkpoint_eval/normalized_constraint_violation": "Constraint Violations",
    }
    
    # Collect results.
    spec = {}
    for d, legend in legend_map.items():
        seed_dirs = os.listdir(os.path.join(config.plot_dir, d))
        spec[legend] = [os.path.join(config.plot_dir, d, sd) for sd in seed_dirs]

    # Collective plot (mean only).
    scalar_stats = plot_from_exps(
        spec,
        out_path=os.path.join(config.plot_dir, "constraint_performance.jpg"),
        scalar_names=["checkpoint_eval/normalized_constraint_violation"],
        title="Learning Curves",
        sub_titles=["Average Normalized Constraint Violations"],
        xlabel="Training Time (s)",
        ylabels=["Constraint Violations"],
        window=None,
        x_num_max=None,
        x_rescale_factor=dt,
        num_std=0,
        use_median_quantile=True,
        cols_per_row=3
    )

    # Save stats to CSV.
    save_stats_to_csv(
        scalar_stats, 
        algo_name_map,  
        scalar_name_map, 
        csv_path=os.path.join(config.plot_dir, "constraint_performance.csv")
    )
    print("Constraint plotting done.")
    

##############################################################################################
#############################   Robustness   #################################################
##############################################################################################

def plot_robustness(config):
    """Gets the plot and csv for robustness (w.r.t system params and action white noise)."""
    if config.param_name in ["pole_length", "pole_mass", "cart_mass"]:
        # with robust_new - pole length 
        legend_map = {
            "ppo": "ppo",
            "ppo_dr_pole_length/low0.1_high1.0": "ppo_dr_0.1-1.0",
            # "ppo_dr_pole_length/low0.1_high0.2": "ppo_dr_0.1-0.2",
            "rap/scale0.1": "rap_scale0.1",
            # "rap/scale0.01": "rap_scale0.01",
        }
    else:
        # with robust_new - input disturbance 
        legend_map = {
            "ppo": "ppo",
            "ppo_dr_pole_length/low0.1_high1.0": "ppo_dr_0.1-1.0",
            # "ppo_dr_pole_length/low0.1_high0.2": "ppo_dr_0.1-0.2",
            "rap/scale0.1": "rap_scale0.1",
            # "rap/scale0.01": "rap_scale0.01",
        }
    
    # e.g. pole_length, act_white
    param_name = config.param_name
    
    # load data (and produce the desired rmse metric)
    data, data_processed = load_eval_stats(
        legend_map, 
        eval_result_dir=config.eval_result_dir, 
        scalar_name="mse", 
        scalar_normalize_by_ep_length=True,
        scalar_postprocess_func=lambda x: np.sqrt(x),
        get_processed_stats=True
    )

    # plot the data, to change size, use `plt.figure(figsize=(18, 6))`
    fig = plt.figure()
    for i, (algo_name, dat) in enumerate(data.items()):
        color = COLORS[i]
        x, y = dat
        # mean across seeds and episodes
        y_mean = y.mean(0).mean(-1)
        plt.plot(x, y_mean, label=algo_name, color=color)
    # plot param values which the algos are trained on
    if hasattr(config, "trained_value"):
        plt.axvline(
            x=float(config.trained_value),
            linewidth=2,
            color='r',
            linestyle="--")
    # figure bookkeeping
    plt.xlabel(param_name)
    plt.ylabel("average rmse")
    plt.title("Robustness w.r.t. {}".format(param_name))
    # legends
    fig.subplots_adjust(bottom=0.2)
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
        # bbox_to_anchor=(0.5, -0.05),
        # bbox_transform=plt.gcf().transFigure,
        fancybox=True,
        shadow=True,
        borderaxespad=0.1,
        ncol=4)
    # save figure
    output_path = os.path.join(config.plot_dir, config.fig_name)
    plt.savefig(output_path)
    plt.show()
    
    # Save stats to CSV.
    scalar_stats = {"Cost": data_processed}
    save_stats_to_csv(
        scalar_stats,
        csv_path=os.path.join(config.plot_dir, "csv/{}_robustness_performance.csv".format(param_name))
    )   
        
    # # Save data to csv & excel. 
    # save_stats_to_excel(
    #     data,
    #     csv_dir=os.path.join(config.plot_dir, "csv")
    # )
    print("Robustness plotting done.")


##############################################################################################
#############################   Mains   ######################################################
##############################################################################################

MAIN_FUNCS = {
    "test_policy": test_policy,
    "test_from_checkpoints": test_from_checkpoints,
    "test_policy_robustness": test_policy_robustness,
    "plot_performance": plot_performance,
    "plot_constraint": plot_constraint,
    "plot_robustness": plot_robustness, 
}


if __name__ == "__main__":
    # Make config.
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="test", help="main function to run.")
    fac.add_argument("--thread", type=int, default=0, help="number of threads to use (set by torch).")
    fac.add_argument("--render", action="store_true", help="if to render in policy test.")
    fac.add_argument("--verbose", action="store_true", help="if to print states & actions in policy test.")
    fac.add_argument("--set_test_seed", action="store_true", help="if to set seed when testing policy.")
    fac.add_argument("--eval_output_dir", type=str, help="folder path to save evaluation results.")
    fac.add_argument("--eval_output_path", type=str, default="test_results.pkl", help="file path to save evaluation results.")

    # testing args
    fac.add_argument("--restore_model_file", type=str, default="model_latest.pt", help="file name to restore the model.")
    fac.add_argument("--fix_env_init", action="store_true", help="if to test policy using env with fixed initial conditions.")
    fac.add_argument("--max_episode_plot", type=int, default=1, help="max number of test episode to plot.")
    fac.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="folder where training checkpoints are saved.")
    fac.add_argument("--set_test_seed_as_training_eval", action="store_true", help="if to set seed same as eval_env during training.")
    fac.add_argument("--checkpoint_eval_dir", type=str, default="checkpoint_eval", help="where to save csv logs evaluated on checkpoints.")

    # plotting args 
    fac.add_argument("--plot_dir", type=str, help="folder path to save the plots.")
    fac.add_argument("--eval_result_dir", type=str, default="eval", help="folder where evaluation files are saved.")
    fac.add_argument("--trained_value", type=str, help="param value used during training")
    fac.add_argument("--fig_name", type=str, default="temp.png", help="output figure name.")
    fac.add_argument("--param_name", type=str, default="pole_length", help="differemt modes for robustness tests.")

    config = fac.merge()
    # System settings.
    if config.thread > 0:
        # E.g. set single thread for less context switching
        torch.set_num_threads(config.thread)
    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))
    func(config)
    