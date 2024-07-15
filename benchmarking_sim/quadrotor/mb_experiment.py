'''A LQR and iLQR example.'''

import os
import sys
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs

script_path = os.path.dirname(os.path.realpath(__file__))

def run(gui=True, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running experiments for model-based methods.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''
    # ALGO = 'ilqr'
    # ALGO = 'gp_mpc'
    # ALGO = 'gpmpc_acados'
    # ALGO = 'mpc'
    ALGO = 'mpc_acados'
    # ALGO = 'lqr'
    # ALGO = 'pid'
    SYS = 'quadrotor_2D_attitude'
    TASK = 'tracking'
    # PRIOR = '300'
    PRIOR = '100'
    agent = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS
    SAFETY_FILTER = None
    # SAFETY_FILTER='linear_mpsc'

    # check if the config file exists
    assert os.path.exists(f'./config_overrides/{SYS}_{TASK}.yaml'), f'./config_overrides/{SYS}_{TASK}.yaml does not exist'
    assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
    if SAFETY_FILTER is None:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    else:
        MPSC_COST='one_step_cost'
        assert ALGO != 'gp_mpc', 'Safety filter only supported for gp_mpc'
        assert os.path.exists(f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--safety_filter', SAFETY_FILTER,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                            f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # merge config
    config = fac.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    
    # Setup safety filter
    if SAFETY_FILTER is not None:
        env_func_filter = partial(make,
                                config.task,
                                **config.task_config)
        safety_filter = make(config.safety_filter,
                            env_func_filter,
                            **config.sf_config)
        safety_filter.reset()

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment.
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()
        # init_state = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        if SAFETY_FILTER is None:
            experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
            experiment.launch_training()
        else:
            safety_filter.learn(env=static_train_env)
            mkdirs(f'{script_path}/models/')
            safety_filter.save(path=f'{script_path}/models/{config.safety_filter}_{SYS}_{TASK}_{PRIOR}.pkl')
            ctrl.reset()
            experiment = BaseExperiment(env=static_env, ctrl=ctrl, safety_filter=safety_filter)

        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        if gui:
            post_analysis(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env)

        # Close environments
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    # # calculate the cost of the trajectory
    # Q = np.diag(config.algo_config.q_lqr)
    # R = np.diag(config.algo_config.r_lqr)
    # cost = 0
    # for i in range(len(all_trajs['obs'][0])-1):
    #     obs = all_trajs['obs'][0][i]
    #     action = all_trajs['action'][0][i]
    #     cost += obs.T @ Q @ obs + action.T @ R @ action
    # cost += all_trajs['obs'][0][-1].T @ Q @ all_trajs['obs'][0][-1]
    # print(f'Total cost of the trajectory: {cost}')

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine iLQR's success.

    Args:
        state_stack (ndarray): The list of observations of iLQR in the latest run.
        input_stack (ndarray): The list of inputs of iLQR in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx)
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    plt.show()


def wrap2pi_vec(angle_vec):
    '''Wraps a vector of angles between -pi and pi.

    Args:
        angle_vec (ndarray): A vector of angles.
    '''
    for k, angle in enumerate(angle_vec):
        while angle > np.pi:
            angle -= np.pi
        while angle <= -np.pi:
            angle += np.pi
        angle_vec[k] = angle
    return angle_vec


if __name__ == '__main__':
    run()
