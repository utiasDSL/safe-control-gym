import os
import sys
sys.path.append(os.getcwd())
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

from eval_utils import * 


##############################################################################################
#############################   Constants   ##################################################
##############################################################################################

# quadrotor controller time interval 
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
        # quad has default 0 for initial states
        init_state = {}
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
    # time and phase trajs for better visualization & plotting
    save_traj_to_csv(config, env, results, os.path.join(eval_output_dir, "eval_trajs"))
    plot_time_traj(env, results, 
                   output_path=os.path.join(eval_output_dir, "time_trajectory.png"), 
                   max_episode_plot=config.max_episode_plot)
    plot_phase_traj(env, results, 
                   output_path=os.path.join(eval_output_dir, "phase_trajectory.png"), 
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


##############################################################################################
#############################   Performance   ##################################################
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
#############################   Constraint   ##################################################
##############################################################################################

def plot_constraint(config):
    """"""
    legend_map = {
        "ppo": "ppo",
        "safe_explorer_ppo/slack0.02": "seppo_0.02",
        # "safe_explorer_ppo/slack0.02_lr0.0001": "seppo_0.02_lr0.0001",
    }
    name_map = {
        "ppo": "PPO",
        "safe_explorer_ppo/slack0.02": "PPO safety_layer slack0.02",
        # "safe_explorer_ppo/slack0.02_lr0.0001": "PPO safety_layer slack0.02 lr 0.0001",
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
#############################   Mains   ######################################################
##############################################################################################

MAIN_FUNCS = {
    "test_policy": test_policy,
    "test_from_checkpoints": test_from_checkpoints,
    "plot_performance": plot_performance,
    "plot_constraint": plot_constraint,
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
    fac.add_argument("--checkpoint_eval_dir", type=str, default="checkpoint_eval", help="where to save csv logs evaluated on checkpoints")
    
    # plotting args
    fac.add_argument("--plot_dir", type=str, help="folder path to save the plots.")
    
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
