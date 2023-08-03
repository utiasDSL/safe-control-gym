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
from matplotlib.patches import Polygon
import csv
from collections import defaultdict
import math 
import glob
import re
from datetime import datetime
from dict_deep import deep_set
import pandas as pd
import seaborn as sns

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
dt = 1.0 / 15.0


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
        config.task_config.seed = env_seed
    else:
        env_seed = None 
    
    # Define function to create task/env.
    if "seed" in config.task_config:
        env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    else:
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
            if k != "ep_lengths":
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
        # "training_results_with_non_opt_hps/temp": "non_opt_hps",
        # "training_results_with_opt_hps/temp": "opt_hps",
        # "training_results_with_opt_hps_eff/temp": "opt_hps_eff",
        # "training_results_with_opt_hps_eff_tradeoff/temp": "opt_hps_tradeoff",
        # "training_results_with_default_hps/temp": "default_hps",
        # "training_results_cutoff_length_as_hp/temp": "opt_hps_inc_cutoff",
        # "training_results_cutoff_length_as_hp_2/temp": "opt_hps_inc_cutoff_2",
        # "No_CVaR/temp": "No_CVaR",
        # "No_CVaR_No_cutoff/temp": "No_CVaR_No_cutoff",
        # "CVaR_0.6/temp": "CVaR_0.6",
        # "CVaR_0.6_dyn/temp": "CVaR_0.6_dyn",
        # "CVaR_0.6_2/temp": "CVaR_0.6_2",
        # "CVaR_0.6_2_dyn/temp": "CVaR_0.6_2_dyn",
        # "CVaR_1/temp": "CVaR_1",
        # "CVaR_1_dyn/temp": "CVaR_1_dyn",
        # "CVaR_0.3/temp": "CVaR_0.3",
        # "CVaR_0.3_dyn/temp": "CVaR_0.3_dyn",
        "prior_150/temp": "prior_150",
        "training_results_with_default_hps/temp": "w/o_prior",
    }
    algo_name_map = {
        # "non_opt_hps": "non_opt_hps",
        # "opt_hps": "opt_hps",
        # "opt_hps_eff": "opt_hps_eff",
        # "opt_hps_tradeoff": "opt_hps_tradeoff",
        # "default_hps": "default_hps",
        # "opt_hps_inc_cutoff": "opt_hps_inc_cutoff",
        # "opt_hps_inc_cutoff_2": "opt_hps_inc_cutoff_2",
        # "No_CVaR": "No_CVaR",
        # "No_CVaR_No_cutoff": "No_CVaR_No_cutoff",
        # "CVaR_0.6": "CVaR_0.6",
        # "CVaR_0.6_dyn": "CVaR_0.6_dyn",
        # "CVaR_0.6_2": "CVaR_0.6_2",
        # "CVaR_0.6_2_dyn": "CVaR_0.6_2_dyn",
        # "CVaR_1": "CVaR_1",
        # "CVaR_1_dyn": "CVaR_1_dyn",
        # "CVaR_0.3": "CVaR_0.3",
        # "CVaR_0.3_dyn": "CVaR_0.3_dyn",
        "piror_150": "prior_150",
        "w/o_prior": "w/o_prior",
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
        # "training_results_with_non_opt_hps/temp": "non_opt_hps",
        # "training_results_with_opt_hps/temp": "opt_hps",
        # "training_results_with_opt_hps_eff/temp": "opt_hps_eff",
        # "training_results_with_opt_hps_eff_tradeoff/temp": "opt_hps_tradeoff",
        # "training_results_with_default_hps/temp": "default_hps",
        # "training_results_cutoff_length_as_hp/temp": "opt_hps_inc_cutoff",
        # "training_results_cutoff_length_as_hp_2/temp": "opt_hps_inc_cutoff_2",
        # "No_CVaR/temp": "No_CVaR",
        # "No_CVaR_No_cutoff/temp": "No_CVaR_No_cutoff",
        # "CVaR_0.6/temp": "CVaR_0.6",
        # "CVaR_0.6_dyn/temp": "CVaR_0.6_dyn",
        # "CVaR_0.6_2/temp": "CVaR_0.6_2",
        # "CVaR_0.6_2_dyn/temp": "CVaR_0.6_2_dyn",
        # "CVaR_1/temp": "CVaR_1",
        # "CVaR_1_dyn/temp": "CVaR_1_dyn",
        # "CVaR_0.3/temp": "CVaR_0.3",
        # "CVaR_0.3_dyn/temp": "CVaR_0.3_dyn",
        "prior_150/temp": "prior_150",
        "training_results_with_default_hps/temp": "w/o_prior",
    }
    algo_name_map = {
        # "non_opt_hps": "non_opt_hps",
        # "opt_hps": "opt_hps",
        # "opt_hps_eff": "opt_hps_eff",
        # "opt_hps_tradeoff": "opt_hps_tradeoff",
        # "default_hps": "default_hps",
        # "opt_hps_inc_cutoff": "opt_hps_inc_cutoff",
        # "opt_hps_inc_cutoff_2": "opt_hps_inc_cutoff_2",
        # "No_CVaR": "No_CVaR",
        # "No_CVaR_No_cutoff": "No_CVaR_No_cutoff",
        # "CVaR_0.6": "CVaR_0.6",
        # "CVaR_0.6_dyn": "CVaR_0.6_dyn",
        # "CVaR_0.6_2": "CVaR_0.6_2",
        # "CVaR_0.6_2_dyn": "CVaR_0.6_2_dyn",
        # "CVaR_1": "CVaR_1",
        # "CVaR_1_dyn": "CVaR_1_dyn",
        # "CVaR_0.3": "CVaR_0.3",
        # "CVaR_0.3_dyn": "CVaR_0.3_dyn",
        "piror_150": "prior_150",
        "w/o_prior": "w/o_prior",
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
#############################   HPO evaluation   #############################################
##############################################################################################

def plot_hpo_eval(config):
    """Gets the plot and csv for performance (in RMSE)."""
    SAMPLER = "TPESampler" # "RandomSampler" or "TPESampler"
    TASK = "cartpole" # "cartpole" or "quadrotor_2D"
    legend_map_s1 = {
        f"hpo_strategy_study_{SAMPLER}/run1_s1": "run1_s1",
        f"hpo_strategy_study_{SAMPLER}/run2_s1": "run2_s1",
        f"hpo_strategy_study_{SAMPLER}/run3_s1": "run3_s1"
    }
    legend_map_s2 = {
        f"hpo_strategy_study_{SAMPLER}/run1_s2": "run1_s2",
        f"hpo_strategy_study_{SAMPLER}/run2_s2": "run2_s2",
        f"hpo_strategy_study_{SAMPLER}/run3_s2": "run3_s2",
    }
    legend_map_s3 = {
        f"hpo_strategy_study_{SAMPLER}/run1_s3": "run1_s3",
        f"hpo_strategy_study_{SAMPLER}/run2_s3": "run2_s3",
        f"hpo_strategy_study_{SAMPLER}/run3_s3": "run3_s3",
    }
    legend_map_s4 = {
        f"hpo_strategy_study_{SAMPLER}/run1_s4": "run1_s4",
        f"hpo_strategy_study_{SAMPLER}/run2_s4": "run2_s4",
        f"hpo_strategy_study_{SAMPLER}/run3_s4": "run3_s4",
    }
    algo_name_map_s1 = {
        "run1_s1": "run1_s1",
        "run2_s1": "run2_s1",
        "run3_s1": "run3_s1",
    }
    algo_name_map_s2 = {
        "run1_s2": "run1_s2",
        "run2_s2": "run2_s2",
        "run3_s2": "run3_s2",
    }
    algo_name_map_s3 = {
        "run1_s3": "run1_s3",
        "run2_s3": "run2_s3",
        "run3_s3": "run3_s3",
    }
    algo_name_map_s4 = {
        "run1_s4": "run1_s4",
        "run2_s4": "run2_s4",
        "run3_s4": "run3_s4",
    }
    scalar_name_map = {
        "checkpoint_eval/normalized_rmse": "Cost",
    }

    legend_map_list = [legend_map_s1, legend_map_s2, legend_map_s3, legend_map_s4]
    algo_name_map_list = [algo_name_map_s1, algo_name_map_s2, algo_name_map_s3, algo_name_map_s4]

    data = {}
    for index, (legend_map, algo_name_map) in enumerate(zip(legend_map_list, algo_name_map_list)):
        # Collect results.
        spec = {}
        for d, legend in legend_map.items():
            seed_dirs = os.listdir(os.path.join(config.plot_dir, d))
            spec[legend] = [os.path.join(config.plot_dir, d, sd) for sd in seed_dirs]

        # Get all stats.
        scalar_stats = load_stats(spec, 
                                scalar_names=["checkpoint_eval/normalized_rmse"], 
                                window=None, 
                                x_num_max=None,
                                x_rescale_factor=dt)
        # Get last step stats
        x_cat, last_step_stats = get_last_stats(scalar_stats)
        
        for i in range(len(x_cat)):
            data[x_cat[i]] = last_step_stats[i]
    
    df = pd.DataFrame(data)
    melted_df = pd.melt(df, var_name='Category_Run', value_name='RMSE Cost')

    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])

    # print the statistics of each category
    print(melted_df.groupby(['Category_Run']).describe())
    print(melted_df.groupby(['Category']).describe())

    plt.figure(figsize=(10, 6))
    # sns.boxplot(x='Category', y='RMSE Cost', hue='Run', data=melted_df)
    # plt.legend(title='Run')
    sns.boxplot(x='Category', y='RMSE Cost', data=melted_df)
    plt.xlabel('Category')
    plt.ylabel('RMSE Cost')
    plt.yscale('log')
    plt.title('HPO Strategy Evaluation')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, "HPO_comparison.jpg"))
    plt.close()
    
    print("HPO evaluation plotting done.")

##############################################################################################
#######################   Hyperparameter sensitivity   #######################################
##############################################################################################
  
def plot_hp_sensitivity(config):
    """Gets the hyperparameter sensitivity plot and csv for final performance."""
    # create xvalue_map ex: {
    #                           activation: {
    #                                           activation/leaky_relu: leaky_relu 
    #                                           activation/relu: relu
    #                                           activation/tanh: tanh
    #                                       }
    #                           ...
    #                       }
    hp_perturb_dir = ['./experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s1/seed8_Jul-19-16-29-41_b40566c/hpo/hyperparameters_139.8787',
                      './experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s2/seed8_Jul-20-11-45-15_b40566c/hpo/hyperparameters_133.0447',
                      './experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s3/seed8_Jul-21-05-26-43_b40566c/hpo/hyperparameters_130.2418',
                      './experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s4/seed8_Jul-21-22-42-04_b40566c/hpo/hyperparameters_132.9955']
    tag = []
    xvalue_maps = {}
    hp_dirs = os.listdir(hp_perturb_dir[0])
    for hp_dir in hp_dirs:
        if os.path.isdir(os.path.join(hp_perturb_dir[0], hp_dir)):
            perturb_dirs = os.listdir(os.path.join(hp_perturb_dir[0], hp_dir))
            xvalue_map = {}
            for perturb_dir in perturb_dirs:  
                xvalue_map[os.path.join(hp_dir, perturb_dir)] = perturb_dir
            xvalue_maps[hp_dir] = xvalue_map

    # Collect results.
    specs = {}
    for hp in hp_dirs:
        if os.path.isdir(os.path.join(hp_perturb_dir[0], hp)):
            spec = {}
            for d, xvalue in xvalue_maps[hp].items():
                _d = os.path.join(d, config.tag)
                seed_dirs = os.listdir(os.path.join(hp_perturb_dir[0], _d))
                spec[xvalue] = [os.path.join(hp_perturb_dir[0], _d, sd) for sd in seed_dirs]
            specs[hp] = spec

    scalar_name_map = {
        "checkpoint_eval/normalized_rmse": "cost",
    }

    data = {}
    for hp, spec in specs.items():
        scalar_stats, x_cat, last_step_stats = plot_from_hp_sens(
                                                                spec,
                                                                out_path=hp_perturb_dir[0],
                                                                scalar_names=["checkpoint_eval/normalized_rmse"],
                                                                title="Hyperparameter Sensitivity Analysis", 
                                                                sub_titles=["Average RMSE"],
                                                                xlabel=hp,
                                                                ylabels="RMSE Cost",
                                                                window=None,
                                                                x_num_max=None,
                                                                x_rescale_factor=dt,
                                                                num_std=0,
                                                                use_median_quantile=True,
                                                                cols_per_row=3
                                                                )
        
        data[x_cat[0]] = last_step_stats[0]

    df = pd.DataFrame(data)
    melted_df = pd.melt(df, var_name='Category_Run', value_name='RMSE Cost')

    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category', y='RMSE Cost', hue='Run', data=melted_df)
    plt.xlabel('Category')
    plt.ylabel('RMSE Cost')
    plt.yscale('log')
    plt.title('HPO Strategy Evaluation')
    plt.legend(title='Run')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, "HPO_comparison.jpg"))
    plt.close()

    print("Hyperparameter sensitivity plotting done.")

##############################################################################################
####################   Hyperparameter Optimization Efforts   #################################
##############################################################################################
  
def plot_hpo_effort(config):
    """Gets the wall clock time and agent runs during hpo."""
    SAMPLER = "RandomSampler" # "RandomSampler" or "TPESampler"
    TASK = "cartpole" # "cartpole" or "quadrotor_2D"
    hpo_folder = f'./experiments/comparisons/ppo/hpo/hpo_strategy_study_{SAMPLER}'
    hpo_strategy_runs = os.listdir(hpo_folder)

    # read std_out.txt to get total agent runs and duration time
    data_time = {}
    data_runs = {}
    for s in hpo_strategy_runs:
        parallel_job_folders = os.listdir(os.path.join(hpo_folder, s))
        duration_time = 0
        total_runs = 0
        for job_folder in parallel_job_folders:
            with open(os.path.join(hpo_folder, s, job_folder, 'std_out.txt'), 'r') as file:
                first_line = file.readline()
                last_line = file.readlines()[-1]
                
                first_timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', first_line)
                last_timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', last_line)
                total_runs_match = re.search(r'Total runs: \d+', last_line)

                first_timestamp = first_timestamp_match.group(0)
                last_timestamp = last_timestamp_match.group(0)
                total_runs = int(total_runs_match.group(0).split(': ')[1])

                # Convert timestamps to datetime objects
                start = datetime.strptime(first_timestamp, '%Y-%m-%d %H:%M:%S,%f')
                end = datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S,%f')

                # Calculate the duration time in hours
                duration_hours = (end - start).total_seconds() / 3600
                

                # check if duration time is larger
                if duration_time < duration_hours:
                    duration_time = duration_hours
                total_runs += int(total_runs_match.group(0).split(': ')[1])

            data_time[s] = {'Duration Time (hours)': duration_time}
            data_runs[s] = {'Total Runs': total_runs}
        
    # add to pandas dataframe
    df = pd.DataFrame(data_time)

    melted_df = pd.melt(df, var_name='Category_Run', value_name='Duration Time (hours)')
    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])
    melted_df.sort_values(by=['Category'])

    plt.figure(figsize=(10, 6))
    # sns.barplot(x='Category', y='Duration Time (hours)', hue='Run', data=melted_df)
    # plt.legend(title='Run')
    sns.barplot(x='Category', y='Duration Time (hours)', data=melted_df, order=['s1', 's2', 's3', 's4'])
    plt.xlabel('Category')
    plt.ylabel('Duration Time (hours)')
    plt.title('HPO Strategy Effort')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, "HPO_time_comparison.jpg"))
    plt.close()

    # add to pandas dataframe
    df = pd.DataFrame(data_runs)

    melted_df = pd.melt(df, var_name='Category_Run', value_name='Total Runs')
    melted_df['Category'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[1])
    melted_df['Run'] = melted_df['Category_Run'].apply(lambda x: x.split('_')[0])

    plt.figure(figsize=(10, 6))
    # sns.barplot(x='Category', y='Total Runs', hue='Run', data=melted_df)
    # plt.legend(title='Run')
    sns.barplot(x='Category', y='Total Runs', data=melted_df, order=['s1', 's2', 's3', 's4'])
    plt.xlabel('Category')
    plt.ylabel('Total Agent Runs')
    plt.yscale('log')
    plt.title('HPO Strategy Effort')
    plt.show()
    plt.savefig(os.path.join(config.plot_dir, "HPO_agent_runs_comparison.jpg"))
    plt.close()

    print("Hyperparameter optimization effort plotting done.")

def plot_hp_sen_comparison(config):
    """Compare hp sensitivity on the same boxplot."""

    hp_perturb_dir = config.hp_perturb_dir
    x_cat = ['Single Run w/o CVaR', 'Dynamic Runs w/ CVaR']
    x_sub_cats = []
    sub_stats = []
    with open(hp_perturb_dir[0], 'r') as f:
        reader = csv.reader(f)
        fixed_runs_data = list(reader)

    for id, row in enumerate(fixed_runs_data):
        if id == 0:
            x_sub_cats.append(row)
        else:
            sub_stats.append(list(map(float, row)))
    stats, cats = [sub_stats], [x_sub_cats]

    x_sub_cats, sub_stats = [], []
    with open(hp_perturb_dir[1], 'r') as f:
        reader = csv.reader(f)
        dyn_runs_data = list(reader)
    
    for id, row in enumerate(dyn_runs_data):
        if id == 0:
            x_sub_cats.append(row)
        else:
            sub_stats.append(list(map(float, row)))
    stats += [sub_stats]
    cats += [x_sub_cats]

    plot_boxplot_from_stats(stats, cats, x_cat, config.plot_dir, file_name='{}.jpg'.format(config.tag))



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
    "plot_hp_sensitivity": plot_hp_sensitivity,
    "plot_hpo_eval": plot_hpo_eval,
    "plot_hp_sen_comparison": plot_hp_sen_comparison,
    "plot_hpo_effort": plot_hpo_effort,
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
    fac.add_argument("--hp_perturb_dir", nargs='+', type=str, help="folder where hp perturb results are saved.")

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
    