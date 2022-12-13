'''This script tests the MPSC safety filter implementation. '''

import os
import shutil
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Cost, Environment


def run(plot=True, training=False, n_episodes=1, n_steps=None, curr_path='.'):
    '''Main function to run MPSC experiments.

    Args:
        plot (bool): Whether to plot the results.
        training (bool): Whether to train the MPSC or load pre-trained values.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
    '''

    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    config.task_config['randomized_init'] = False
    if config.algo in ['ppo', 'sac']:
        config.task_config['cost'] = Cost.RL_REWARD
        config.algo_config['training'] = False
    else:
        config.task_config['cost'] = Cost.QUADRATIC
        config.task_config['normalized_rl_action_space'] = False

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
                    output_dir=curr_path+'/temp')

    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        model_dir = os.path.dirname(os.path.abspath(__file__))+'/models'
        task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
        ctrl.load(os.path.join(model_dir, f'{config.algo}_model_{system}_{task}.pt'))

        # Remove temporary files and directories
        shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp', ignore_errors=True)

    # Run without safety filter
    experiment = BaseExperiment(env, ctrl)
    results, uncert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    elapsed_time_uncert = results['timestamp'][0][-1] - results['timestamp'][0][0]

    # Setup MPSC.
    config.task_config['normalized_rl_action_space'] = False
    env_func_filter = partial(make,
                       config.task,
                       **config.task_config)
    safety_filter = make(config.safety_filter,
                env_func_filter,
                **config.sf_config)
    safety_filter.reset()

    if training is True:
        train_env = env_func(randomized_init=True,
                             init_state=None,
                             cost='quadratic',
                             normalized_rl_action_space=False,
                             disturbance=None,
                            )
        safety_filter.learn(env=train_env)
        safety_filter.save(path=f'{curr_path}/models/{config.safety_filter}_{system}.pkl')
    else:
        safety_filter.load(path=f'{curr_path}/models/{config.safety_filter}_{system}.pkl')

    ctrl.reset()

    # Run with safety filter
    experiment = BaseExperiment(env, ctrl, safety_filter=safety_filter)
    certified_results, cert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    ctrl.close()
    mpsc_results = certified_results['safety_filter_data'][0]
    safety_filter.close()

    elapsed_time_cert = results['timestamp'][0][-1] - results['timestamp'][0][0]

    corrections = mpsc_results['correction'][0] > 1e-6
    corrections = np.append(corrections, False)

    if plot is True:
        if config.task == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif config.task == Environment.QUADROTOR:
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2

        _, ax = plt.subplots()
        ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='Uncertified')
        ax.plot(certified_results['obs'][0][:,graph1_1], certified_results['obs'][0][:,graph1_2],'.-', label='Certified')
        ax.plot(certified_results['obs'][0][corrections, graph1_1], certified_results['obs'][0][corrections, graph1_2], 'r.', label='Modified')
        ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        if config.task == Environment.CARTPOLE:
            theta_constraint = config.task_config['constraints'][0].upper_bounds[2]
        elif config.task == Environment.QUADROTOR:
            theta_constraint = config.task_config['constraints'][0].upper_bounds[4]
        ax.axvline(x=-theta_constraint, color='k', lw=2, label='Limit')
        ax.axvline(x=theta_constraint, color='k', lw=2)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
            _, ax2 = plt.subplots()
            ax2.plot(np.linspace(0, 20, certified_results['obs'][0].shape[0])[1:], safety_filter.env.X_GOAL[:,0],'g--', label='Reference')
            ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:,0],'r--', label='Uncertified')
            ax2.plot(np.linspace(0, 20, certified_results['obs'][0].shape[0]), certified_results['obs'][0][:,0],'.-', label='Certified')
            ax2.plot(np.linspace(0, 20, certified_results['obs'][0].shape[0])[corrections], certified_results['obs'][0][corrections, 0], 'r.', label='Modified')
            ax2.set_xlabel(r'Time')
            ax2.set_ylabel(r'X')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(results['obs'][0][:,1], results['obs'][0][:,3],'r--', label='Uncertified')
            ax2.plot(certified_results['obs'][0][:,1], certified_results['obs'][0][:,3],'.-', label='Certified')
            ax2.plot(certified_results['obs'][0][corrections, 1], certified_results['obs'][0][corrections, 3], 'r.', label='Modified')
            ax2.set_xlabel(r'x_dot')
            ax2.set_ylabel(r'z_dot')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:,graph3_1], results['obs'][0][:,graph3_2],'r--', label='Uncertified')
        ax3.plot(certified_results['obs'][0][:,graph3_1], certified_results['obs'][0][:,graph3_2],'.-', label='Certified')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(safety_filter.env.X_GOAL[:,0], safety_filter.env.X_GOAL[:,2],'g--', label='Reference')
        ax3.plot(certified_results['obs'][0][corrections, graph3_1], certified_results['obs'][0][corrections, graph3_2], 'r.', label='Modified')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        _, ax_act = plt.subplots()
        if config.task == Environment.CARTPOLE:
            ax_act.plot(certified_results['current_physical_action'][0][:], 'b-', label='Certified Input')
            ax_act.plot(mpsc_results['uncertified_action'][0][:], 'r--', label='Attempted Input')
            ax_act.plot(results['current_physical_action'][0][:], 'g--', label='Uncertified Input')
        else:
            ax_act.plot(certified_results['current_physical_action'][0][:, 0], 'b-', label='Certified Input 1')
            ax_act.plot(certified_results['current_physical_action'][0][:, 1], 'b--', label='Certified Input 2')
            ax_act.plot(mpsc_results['uncertified_action'][0][:, 0], 'r-', label='Attempted Input 1')
            ax_act.plot(mpsc_results['uncertified_action'][0][:, 1], 'r--', label='Attempted Input 2')
            ax_act.plot(results['current_physical_action'][0][:, 0], 'g-', label='Uncertified Input 1')
            ax_act.plot(results['current_physical_action'][0][:, 1], 'g--', label='Uncertified Input 2')
        ax_act.legend()
        ax_act.set_title('Input comparison')
        ax_act.set_xlabel('Step')
        ax_act.set_ylabel('Input')
        ax_act.set_box_aspect(0.5)

        print(f'Total Uncertified Time: {elapsed_time_uncert}s')
        print(f'Total Certified Time: {elapsed_time_cert}s')
        print('Number of Corrections: ', np.sum(corrections))
        print('Sum of Corrections: ', np.linalg.norm(mpsc_results['correction'][0]))
        print('Max Correction: ', np.max(np.abs(mpsc_results['correction'][0])))
        print('Number of Feasible Iterations: ', np.sum(mpsc_results['feasible'][0]))
        print('Total Number of Iterations: ', uncert_metrics['average_length'])
        print('Total Number of Certified Iterations: ', cert_metrics['average_length'])
        print('Number of Violations: ', uncert_metrics['average_constraint_violation'])
        print('Number of Certified Violations: ', cert_metrics['average_constraint_violation'])
        print('RMSE Uncertified: ', uncert_metrics['average_rmse'])
        print('RMSE Certified: ', cert_metrics['average_rmse'])

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    run()
