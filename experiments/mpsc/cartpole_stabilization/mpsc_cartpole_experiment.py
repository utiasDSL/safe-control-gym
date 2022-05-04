"""This script tests the MPSC safety filter implementation

"""
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym import safety_filters
from safe_control_gym.envs import disturbances
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory


def main():
    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()
    
    # Setup controller.
    ctrl = make(config.algo,
                    env_func,
                    **config.algo_config)
    
    # Load state_dict from trained.
    model_dir = os.path.dirname(os.path.abspath(__file__))+'/models'
    ctrl.load(os.path.join(model_dir, f'{config.algo}_model_cartpole.pt'))  # Show violation.
    
    # Remove temporary files and directories
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp', ignore_errors=True)
    
    # Run without safety filter
    iterations = 150
    _, results = ctrl.run(env=env, num_iterations=iterations)
    ctrl.reset()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                env_func,
                **config.sf_config)
    safety_filter.reset()

    # train_env = env_func(randomized_init=True, init_state=None, disturbances=None) # training without disturbances
    train_env = env_func(randomized_init=True, init_state=None) # training with disturbances
    safety_filter.learn(env=train_env)

    ctrl.safety_filter = safety_filter

    # Start a timer.
    START = time.time()
    
    # Run with safety filter
    _, certified_results = ctrl.run(env=env, num_iterations=iterations)
    ctrl.close()
    safety_filter.close()

    elapsed_time = time.time() - START
    print(f"Total MPSC Time: {elapsed_time}s")

    # # Plot results
    # fig_obs, ax_obs = plt.subplots()  
    # ax_obs.plot(certified_results.obs[:, 0], certified_results.obs[:, 2], '.-', label='Certified')
    # ax_obs.plot(results.obs[:, 0], results.obs[:, 2], 'r--', label='Uncertified')
    # ax_obs.plot(certified_results.obs[certified_results.corrections>1e-6, 0], certified_results.obs[certified_results.corrections>1e-6, 2], 'r.', label='Modified')
    # ax_obs.legend()
    # ax_obs.set_title('State Space')
    # ax_obs.set_xlabel(r'$x$')
    # ax_obs.set_ylabel(r'$\theta$')
    # ax_obs.set_box_aspect(0.5)
    
    # fig_act, ax_act = plt.subplots()
    # ax_act.plot(certified_results.action[:], 'b-', label='Certified Inputs')
    # ax_act.plot(results.action[:], 'r--', label='Uncertified Input')
    # ax_act.legend()
    # ax_act.set_title('Input comparison')
    # ax_act.set_xlabel('Step')
    # ax_act.set_ylabel('Input')
    # ax_act.set_box_aspect(0.5)
    
    fig, ax = plt.subplots()
    ax.plot(certified_results.obs[:,2], certified_results.obs[:,3],'.-', label='Certified')
    corrections = certified_results.corrections>1e-6
    print("Number of Corrections: ", np.sum(corrections))
    print("Total Number of Iterations: ", corrections.shape[0])
    # corrections = list(safety_filter_results.violation) + [False]
    ax.plot(certified_results.obs[corrections, 2], certified_results.obs[corrections, 3], 'r.', label='Modified')
    ax.plot(results.obs[:, 2], results.obs[:, 3], 'r--', label='Uncertified')
    ax.axvline(x=-0.2, color='k', lw=2, label='Limit')
    ax.scatter(results.obs[0, 2], results.obs[0, 3], color='g', marker='o', s=100, label='Initial State')
    ax.axvline(x=0.2, color='k', lw=2)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_box_aspect(0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    # plt.savefig('./temp-data/mpsc.png', dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
