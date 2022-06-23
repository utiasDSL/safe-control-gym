"""This script tests the MPSC safety filter implementation

"""
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Cost, Environment


def run(plot=True, max_steps=300, curr_path='.'):
    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    if config.algo in ['ppo', 'sac', 'rarl']:
        config.task_config['cost'] = Cost.RL_REWARD
    else:
        config.task_config['cost'] = Cost.QUADRATIC
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    
    # Setup controller.
    ctrl = make(config.algo,
                    env_func,
                    **config.algo_config,
                    output_dir=curr_path+'/temp')
    
    if config.algo in ['ppo', 'sac', 'rarl']:
        # Load state_dict from trained.
        model_dir = os.path.dirname(os.path.abspath(__file__))+'/models'
        ctrl.load(os.path.join(model_dir, f'{config.algo}_model_{config.task}.pt'))
        
        # Remove temporary files and directories
        shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp', ignore_errors=True)
    
    # Start a timer.
    START = time.time()

    # Run without safety filter
    if config.algo in ['ppo', 'sac', 'rarl']:
        results = ctrl.run(max_steps=max_steps, n_episodes=1)
        results = results['ep_results'][0]
    else:
        results = ctrl.run(max_steps=max_steps)
    elapsed_time_uncert = time.time() - START

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                env_func,
                **config.sf_config)
    safety_filter.reset()
    
    # train_env = env_func(randomized_init=True, init_state=None, cost='quadratic', disturbance=None)
    # train_env = env_func(randomized_init=True, init_state=None, cost='quadratic')
    # safety_filter.learn(env=train_env)
    safety_filter.load(path_to_P=f'{curr_path}/P_{config.task}.npy')

    ctrl.safety_filter = safety_filter
    ctrl.reset()

    # Start a timer.
    START = time.time()
    
    # Run with safety filter
    if config.algo in ['ppo', 'sac', 'rarl']:
        certified_results = ctrl.run(max_steps=max_steps, n_episodes=1)
        certified_results = certified_results['ep_results'][0]
    else:
        certified_results = ctrl.run(max_steps=max_steps)
    ctrl.close()
    safety_filter.close()
    safety_filter.close_results_dict()
    safety_filter_results = safety_filter.results_dict

    elapsed_time_cert = time.time() - START
    
    corrections = certified_results.corrections>1e-6
    num_violations = sum([results.info[i][0]['constraint_violation'] for i in range(len(results.info))])
    num_certified_violations = sum([certified_results.info[i][0]['constraint_violation'] for i in range(len(certified_results.info))])

    if plot:
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
        ax.plot(results.obs[:, graph1_1], results.obs[:, graph1_2], 'r--', label='Uncertified')
        ax.plot(certified_results.obs[:,graph1_1], certified_results.obs[:,graph1_2],'.-', label='Certified')
        ax.plot(certified_results.obs[corrections, graph1_1], certified_results.obs[corrections, graph1_2], 'r.', label='Modified')
        ax.scatter(results.obs[0, graph1_1], results.obs[0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        if config.task == Environment.CARTPOLE:
            ax.axvline(x=-0.2, color='k', lw=2, label='Limit')
            ax.axvline(x=0.2, color='k', lw=2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
            _, ax2 = plt.subplots()
            ax2.plot(np.linspace(0, 20, certified_results.obs.shape[0])[1:], safety_filter.env.X_GOAL[:,0],'.-', label='Reference')
            ax2.plot(np.linspace(0, 20, results.obs.shape[0]), results.obs[:,0],'.-', label='Uncertified')
            ax2.plot(np.linspace(0, 20, certified_results.obs.shape[0]), certified_results.obs[:,0],'.-', label='Certified')
            ax2.plot(np.linspace(0, 20, certified_results.obs.shape[0])[corrections], certified_results.obs[corrections, 0], 'r.', label='Modified')
            ax2.set_xlabel(r"Time")
            ax2.set_ylabel(r"X")
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(results.obs[:,1], results.obs[:,3],'.-', label='Uncertified')
            ax2.plot(certified_results.obs[:,1], certified_results.obs[:,3],'.-', label='Certified')
            ax2.plot(certified_results.obs[corrections, 1], certified_results.obs[corrections, 3], 'r.', label='Modified')
            ax2.set_xlabel(r"x_dot")
            ax2.set_ylabel(r"z_dot")
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(results.obs[:,graph3_1], results.obs[:,graph3_2],'r--', label='Uncertified')
        ax3.plot(certified_results.obs[:,graph3_1], certified_results.obs[:,graph3_2],'.-', label='Certified')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(safety_filter.env.X_GOAL[:,0], safety_filter.env.X_GOAL[:,2],'g--', label='Reference')
        ax3.plot(certified_results.obs[corrections, graph3_1], certified_results.obs[corrections, graph3_2], 'r.', label='Modified')
        ax3.scatter(results.obs[0, graph3_1], results.obs[0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r"X")
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r"Vel")
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r"Z")
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        print(f"Total Uncertified Time: {elapsed_time_uncert}s")
        print(f"Total Certified Time: {elapsed_time_cert}s")
        print("Number of Corrections: ", np.sum(corrections))
        print("Sum of Corrections: ", np.linalg.norm(certified_results.corrections))
        print("Max Correction: ", np.max(np.abs(certified_results.corrections)))
        print("Number of Feasible Iterations: ", np.sum(safety_filter_results.feasible))
        print("Total Number of Iterations: ", results.obs.shape[0])
        print("Total Number of Certified Iterations: ", certified_results.obs.shape[0])
        print("Number of Violations: ", num_violations)
        print("Number of Certified Violations: ", num_certified_violations)

        plt.tight_layout()
        # plt.savefig('./temp-data/mpsc.png', dpi=500)
        plt.show()


if __name__ == "__main__":
    run()
