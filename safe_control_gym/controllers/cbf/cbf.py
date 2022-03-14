"""Control barrier function (CBF) main

Examples
    check CBF candidate on cartpole balance:

        python3 cbf.py --func is_cbf --algo cbf --task cartpole --overrides ./cbf_qp_verify.yaml

    train on cartpole balance:

        python3 cbf.py --func train --algo cbf --task cartpole --overrides ./cbf_qp_learn.yaml

    test on cartpole balance:

        python3 cbf.py --func test --algo cbf --task cartpole --overrides ./cbf_qp_test.yaml

"""

from functools import partial
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config


def plot_is_cbf(infeasible_states, maximum_states):
    # Plot feasible and infeasible points

    state_ids = {"x_pos": 0, "x_dot": 1, "theta": 2, "theta_dot": 3}
    max_states = {}
    for i, state_id in enumerate(state_ids.keys()):
        max_states[state_id] = maximum_states[i]

    phi = np.linspace(0, 2 * np.pi, num=120)

    plots = [["x_pos", "theta"], ["theta", "theta_dot"], ["x_pos", "x_dot"]]

    for i, plot in enumerate(plots):
        if plot[0] == "theta" and plot[1] == "theta_dot":
            plt.plot(max_states[plot[0]] * np.sin(phi), max_states[plot[1]] * np.cos(phi), label="superlevel set")
        plt.xlabel(plot[0])
        plt.ylabel(plot[1])

        if len(infeasible_states) > 0:
            for index, infeasible_state in enumerate(infeasible_states):
                if index == 0:
                    plt.plot(infeasible_state[state_ids[plot[0]]], infeasible_state[state_ids[plot[1]]], "rx",
                             label="infeasible state")
                else:
                    plt.plot(infeasible_state[state_ids[plot[0]]], infeasible_state[state_ids[plot[1]]], "rx")

        plt.legend()
        plt.show()


def plot_test(stats_buffer, maximum_states):
    state_ids = {"state/x_pos": 0, "state/x_dot": 1, "state/theta": 2, "state/theta_dot": 3}
    max_states = {}
    for i, state_id in enumerate(state_ids.keys()):
        max_states[state_id] = maximum_states[i]

    plots = [["t", ["action/safe_input", "action/unsafe_input", "action/applied_input"]],
             ["state/theta", "state/theta_dot"],
             ["state/x_pos", "state/x_dot"],
             ["state/x_pos", "state/theta"]]

    t = range(len(stats_buffer[plots[0][1][0]]))
    print("Num time steps:", len(t))

    phi = np.linspace(0, 2 * np.pi, num=120)
    line_styles = ["m-", "c-", "k--"]

    for plot in plots:
        if plot[0] == "t":
            plot_x = t
            plt.xlabel("step count")
            plt.ylabel("control input")
        else:
            plot_x = stats_buffer[plot[0]]
            if plot[0] == "state/theta" and plot[1] == "state/theta_dot":
                plt.plot(max_states[plot[0]] * np.sin(phi), max_states[plot[1]] * np.cos(phi), "b-",
                         label="superlevel set")
            plt.xlabel(plot[0])
            plt.ylabel(plot[1])
        if isinstance(plot[1], list):
            for index, plot_y_id in enumerate(plot[1]):
                plt.plot(plot_x, stats_buffer[plot_y_id], line_styles[index], label=plot_y_id)
        else:
            plot_y = stats_buffer[plot[1]]
            plt.plot(plot_x, plot_y)
            plt.plot(plot_x[0], plot_y[0], 'gx', label="start")
            plt.plot(plot_x[-1], plot_y[-1], 'rx', label="end")

        plt.legend()
        plt.show()


def is_cbf(config):
    """
    Check if the provided CBF candidate is a CBF for the true system and the a priori system.
    """

    # Evaluation setup
    set_device_from_config(config)

    is_cbf = [False] * 2
    maximum_states = [config.algo_config.x_pos_max,
                      config.algo_config.x_vel_max,
                      config.algo_config.theta_max,
                      config.algo_config.theta_dot_max]

    # Check CBF for true system and the a priori system
    for i in range(2):
        if i == 0:
            print("--------------------------------------------------------")
            print("1. Check provided CBF candidate for the a priori system.")
            print("--------------------------------------------------------")
        elif i == 1:
            print("--------------------------------------------------------")
            print("2. Check provided CBf candidate for the true system.")
            print("--------------------------------------------------------")

        # Define function to create task/env
        env_func = partial(make,
                           config.task,
                           output_dir=config.output_dir,
                           # prior_prop=config.task_config.prior_prop,
                           **config.task_config)

        # Create the controller/control_agent.
        control_agent = make(config.algo,
                             env_func,
                             training=False,
                             checkpoint_path=os.path.join(config.output_dir,
                                                          "model_latest.pt"),
                             output_dir=config.output_dir,
                             device=config.device,
                             seed=config.seed,
                             **config.algo_config)

        control_agent.reset()

        num_points = config.algo_config.num_points
        tolerance = config.algo_config.tolerance

        is_cbf[i], infeasible_states = control_agent.is_cbf(num_points=num_points, tolerance=tolerance)
        control_agent.close()

        plot_is_cbf(infeasible_states, maximum_states)

        if i == 0:
            # Switch to the true system for the next check
            config.task_config.prior_prop = None

    if is_cbf[0] and is_cbf[1]:
        print("------------------------------------------------------------------------------------------")
        print("The provided CBF candidate is potentially a CBF for both the true and the a priori system.")
        print("------------------------------------------------------------------------------------------")
    else:
        print("------------------------------------------------------------------------------------------")
        print("The provided CBF candidate is NOT a CBF for both the true and the a priori system.")
        print("------------------------------------------------------------------------------------------")


def train(config):
    """General training template.
    
    Usage:
        * to start training, use with `--func train`.
        * to restore from a previous training, additionally use `--restore {dir_path}` 
            where `dir_path` is the output folder from previous training.  

    """
    # Evaluation setup
    set_device_from_config(config)

    maximum_states = [config.algo_config.x_pos_max,
                      config.algo_config.x_vel_max,
                      config.algo_config.theta_max,
                      config.algo_config.theta_dot_max]

    # Define function to create task/env
    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config)

    unsafe_control_agent = make(config.algo_config["unsafe_controller"],
                                env_func,
                                training=False,
                                checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                                output_dir=config.output_dir,
                                device=config.device,
                                seed=config.seed,
                                **config.algo_config)

    config.algo_config["unsafe_controller"] = unsafe_control_agent

    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir,
                                                      "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)

    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, "model_latest.pt"))

    control_agent.learn()
    control_agent.close()


def test_policy(config):
    """Run the (trained) policy/controller for evaluation.
    
    Usage
        * use with `--func test`.
        * to test policy from a trained model checkpoint, additionally use 
            `--restore {dir_path}` where `dir_path` is folder to the trained model.
        * to test un-trained policy (e.g. non-learning based), use as it is.

    """
    # Evaluation setup
    set_device_from_config(config)

    maximum_states = [config.algo_config.x_pos_max,
                      config.algo_config.x_vel_max,
                      config.algo_config.theta_max,
                      config.algo_config.theta_dot_max]

    # Define function to create task/env
    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config)

    unsafe_control_agent = make(config.algo_config["unsafe_controller"],
                                env_func,
                                training=False,
                                checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                                output_dir=config.output_dir,
                                device=config.device,
                                seed=config.seed,
                                **config.algo_config)

    config.algo_config["unsafe_controller"] = unsafe_control_agent

    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir,
                                                      "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)

    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, "model_latest.pt"))

    stats_buffer = control_agent.run(logging=True)
    control_agent.close()

    plot_test(stats_buffer, maximum_states)


MAIN_FUNCS = {"train": train, "test": test_policy, "is_cbf": is_cbf}

if __name__ == "__main__":
    # Make config
    fac = ConfigFactory()
    fac.add_argument("--func",
                     type=str,
                     default="train",
                     help="main function to run.")
    fac.add_argument("--thread",
                     type=int,
                     default=0,
                     help="number of threads to use (set by torch).")
    fac.add_argument("--render",
                     action="store_true",
                     help="if to render in policy test.")
    fac.add_argument("--verbose",
                     action="store_true",
                     help="if to print states & actions in policy test.")
    fac.add_argument("--use_adv",
                     action="store_true",
                     help="if to evaluate against adversary.")
    fac.add_argument("--eval_output_path",
                     type=str,
                     default="test_results.pkl",
                     help="file path to save evaluation results.")
    config = fac.merge()

    # system settings
    if config.thread > 0:
        # e.g. set single thread for less context switching
        torch.set_num_threads(config.thread)

    # Execute
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))

    if "cartpole" not in config.task:
        raise Exception("Task {} not supported.".format(config.task))

    func(config)
