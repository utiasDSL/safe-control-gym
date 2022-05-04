"""This script tests the MPSC safety filter implementation

"""
import os
import time
import shutil
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory


def main():
    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    if config.algo in ['ppo', 'sac']:
        config.task_config['cost'] = 'rl_reward'
    else:
        config.task_config['cost'] = 'quadratic'
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    
    # Setup controller.
    ctrl = make(config.algo,
                    env_func,
                    **config.algo_config)
    
    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        model_dir = os.path.dirname(os.path.abspath(__file__))+'/models'
        ctrl.load(os.path.join(model_dir, f'{config.algo}_model_quadrotor.pt'))  # Show violation.
        
        # Remove temporary files and directories
        shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp', ignore_errors=True)
    
    # Run without safety filter
    iterations = 250
    if config.algo in ['ppo', 'sac']:
        _, results = ctrl.run(num_iterations=iterations)
    else:
        results = ctrl.run(num_iterations=iterations)
    ctrl.reset()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                env_func,
                **config.sf_config)
    safety_filter.reset()
    train_env = env_func(randomized_init=True, init_state=None, cost='quadratic')
    safety_filter.learn(env=train_env)
    # safety_filter.load(path='P.npy')

    ctrl.safety_filter = safety_filter

    # Start a timer.
    START = time.time()
    
    # Run with safety filter
    if config.algo in ['ppo', 'sac']:
        _, certified_results = ctrl.run(num_iterations=iterations)
    else:
        certified_results = ctrl.run(num_iterations=iterations)
    ctrl.close()
    safety_filter.close()

    elapsed_time = time.time() - START
    print(f"Total MPSC Time: {elapsed_time}s")

    # Plot results
    fig_obs, ax = plt.subplots()  
    ax.axhline(y=0, color='k', lw=2, label='Limit')
    ax.plot(certified_results.obs[:, 0], certified_results.obs[:, 2], '.-', label='Certified')
    ax.plot(results.obs[:, 0], results.obs[:, 2], 'r--', label='Uncertified')
    ax.plot(certified_results.obs[certified_results.corrections>1e-6, 0], certified_results.obs[certified_results.corrections>1e-6, 2], 'r.', label='Modified')
    ax.legend()
    ax.set_title('State Space')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_box_aspect(0.5)

    plt.tight_layout()
    plt.savefig('./temp-data/mpsc_quad.png', dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
