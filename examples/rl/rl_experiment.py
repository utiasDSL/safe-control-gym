'''This script tests the RL implementation. '''

import shutil
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Environment


def run(gui=True, n_episodes=1, n_steps=None, curr_path='.'):
    '''Main function to run RL experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.

    Returns:
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        results (dict): The results of the experiment.
        metrics (dict): The metrics of the experiment.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')

    # Load state_dict from trained.
    ctrl.load(f'{curr_path}/models/{config.algo}/{config.algo}_model_{system}_{task}.pt')

    # Remove temporary files and directories
    shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)

    # Run experiment
    experiment = BaseExperiment(env, ctrl)
    results, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    ctrl.close()

    if gui is True:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4

        _, ax = plt.subplots()
        ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='RL Trajectory')
        ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
            _, ax2 = plt.subplots()
            ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:, 0], 'r--', label='RL Trajectory')
            ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), env.X_GOAL[:, 0], 'b', label='Reference')
            ax2.set_xlabel(r'Time')
            ax2.set_ylabel(r'X')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(results['obs'][0][:, graph3_1 + 1], results['obs'][0][:, graph3_2 + 1], 'r--', label='RL Trajectory')
            ax2.set_xlabel(r'x_dot')
            ax2.set_ylabel(r'z_dot')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='RL Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(env.X_GOAL[:, graph3_1], env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return env.X_GOAL, results, metrics


if __name__ == '__main__':
    run()
