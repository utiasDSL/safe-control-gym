"""Template hyperparameter optimization/hyperparameter evaluation script.

"""
import os
from functools import partial

import yaml

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Environment, Task

from safe_control_gym.hyperparameters.hpo import HPO
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config


def hpo(config):
    """Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.

    """

    # Experiment setup.
    if config.hpo_config.hpo:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # HPO
    hpo = HPO(config.algo,
              config.task,
              config.sampler,
              config.load_study,
              config.output_dir,
              config.task_config,
              config.hpo_config,
              **config.algo_config)

    if config.hpo_config.hpo:
        hpo.hyperparameter_optimization()
        print('Hyperparameter optimization done.')


def train(config):
    """Training for a given set of hyperparameters.

    Usage:
        * to start training, use with `--func train`.

    """
    # Override algo_config with given yaml file
    if config.opt_hps == '':
        # if no opt_hps file is given
        pass
    else:
        # if opt_hps file is given
        with open(config.opt_hps, 'r') as f:
            opt_hps = yaml.load(f, Loader=yaml.FullLoader)
        for hp in opt_hps:
            if isinstance(config.algo_config[hp], list) and not isinstance(opt_hps[hp], list):
                config.algo_config[hp] = [opt_hps[hp]] * len(config.algo_config[hp])
            else:
                config.algo_config[hp] = opt_hps[hp]
    # Experiment setup.
    if config.plot_best is False:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    # Note:
    # eval_env will take config.seed * 111 as its seed
    # env will take config.seed as its seed
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()

    eval_env = env_func(seed=config.seed * 111)

    if config.plot_best:
        control_agent.load('examples/hpo/results/2D/seed6_May-15-11-07-56_v0.5.0-611-gce9662f/model_best.pt')
        experiment = BaseExperiment(eval_env, control_agent)
    else:
        experiment = BaseExperiment(eval_env, control_agent)
        experiment.launch_training()
    results, metrics = experiment.run_evaluation(n_episodes=config.n_episodes, n_steps=None, done_on_max_steps=True)
    control_agent.close()

    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
        if config.task_config.quad_type == 4:
            system = 'quadrotor_2D'
    else:
        system = config.task

    if True:
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
        
        if config.task_config.quad_type != 4:
            _, ax = plt.subplots()
            ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='Agent Trajectory')
            ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
            ax.set_xlabel(r'$\theta$')
            ax.set_ylabel(r'$\dot{\theta}$')
            ax.set_box_aspect(0.5)
            ax.legend(loc='upper right')
            # save the plot
            plt.savefig(os.path.join(config.output_dir, 'trajectory_theta_theta_dot.png'))

            if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
                _, ax2 = plt.subplots()
                ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:, 0], 'r--', label='Agent Trajectory')
                ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), eval_env.X_GOAL[:, 0], 'b', label='Reference')
                ax2.set_xlabel(r'Time')
                ax2.set_ylabel(r'X')
                ax2.set_box_aspect(0.5)
                ax2.legend(loc='upper right')
                # save the plot
                plt.savefig(os.path.join(config.output_dir, 'trajectory_time_x.png'))
            elif config.task == Environment.QUADROTOR:
                _, ax2 = plt.subplots()
                ax2.plot(results['obs'][0][:, graph3_1 + 1], results['obs'][0][:, graph3_2 + 1], 'r--', label='Agent Trajectory')
                ax2.set_xlabel(r'x_dot')
                ax2.set_ylabel(r'z_dot')
                ax2.set_box_aspect(0.5)
                ax2.legend(loc='upper right')
                # save the plot
                plt.savefig(os.path.join(config.output_dir, 'trajectory_x_dot_z_dot.png'))

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='Agent Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(eval_env.X_GOAL[:, graph3_1], eval_env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        # save the plot
        plt.savefig(os.path.join(config.output_dir, 'trajectory_x.png'))

    # save to pickle
    with open(os.path.join(config.output_dir, 'metrics.pkl'), 'wb') as f:
        import pickle
        pickle.dump(metrics, f)
    
    return eval_env.X_GOAL, results, metrics


MAIN_FUNCS = {'hpo': hpo, 'train': train}


if __name__ == '__main__':

    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    fac.add_argument('--plot_best', type=bool, default=False, help='plot best agent trajectory.')
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception('Main function {} not supported.'.format(config.func))
    func(config)
