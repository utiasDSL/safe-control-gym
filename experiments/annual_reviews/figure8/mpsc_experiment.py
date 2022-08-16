'''This script runs the MPSC experiment in our Annual Reviews article.

See Figure 8 in https://arxiv.org/pdf/2108.06266.pdf.
'''

from functools import partial
import pickle

import os
import shutil
import matplotlib.pyplot as plt

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import set_random_state


def main():
    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    uncertified_env = env_func()
    certified_env = env_func()

    # Setup PPO controller.
    ctrl = make(config.algo,
                    env_func,
                    **config.algo_config)

    # Load state_dict from trained PPO.
    ppo_model_dir = os.path.dirname(os.path.abspath(__file__))+'/unsafe_ppo_model'
    ctrl.load(os.path.join(ppo_model_dir,'unsafe_ppo_model_30000.pt'))  # Show violation.

    # Remove temporary files and directories
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp')

    # Run without safety filter
    max_steps = 30
    with open('./unsafe_ppo_model/state1.pkl', 'rb') as f:
        state1 = pickle.load(f)
    set_random_state(state1)
    results = ctrl.run(env=uncertified_env, max_steps=max_steps, n_episodes=1, use_step=True)
    results = results['ep_results'][0]
    uncertified_env.close()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                env_func,
                **config.sf_config)
    safety_filter.reset()

    train_env = env_func(init_state=None)
    safety_filter.learn(env=train_env)

    ctrl.safety_filter = safety_filter
    ctrl.reset()

    # Run with safety filter
    with open('./unsafe_ppo_model/state0.pkl', 'rb') as f:
        state0 = pickle.load(f)
    set_random_state(state0)
    certified_results = ctrl.run(env=certified_env, max_steps=max_steps, n_episodes=1, use_step=True)
    certified_results = certified_results['ep_results'][0]
    certified_env.close()
    ctrl.close()
    safety_filter.close_results_dict()
    mpsc_results = safety_filter.results_dict
    safety_filter.close()

    # Plot Results
    _, ax_obs = plt.subplots()
    ax_obs.plot(certified_results.obs[:, 0], certified_results.obs[:, 2], '.-', label='Certified')
    ax_obs.plot(results.obs[:10, 0], results.obs[:10, 2], 'r--', label='Uncertified')
    ax_obs.plot(certified_results.obs[certified_results.corrections>1e-6, 0], certified_results.obs[certified_results.corrections>1e-6, 2], 'r.', label='Modified')
    ax_obs.legend()
    ax_obs.set_title('State Space')
    ax_obs.set_xlabel(r'$x$')
    ax_obs.set_ylabel(r'$\theta$')
    ax_obs.set_box_aspect(0.5)

    _, ax_act = plt.subplots()
    ax_act.plot(certified_results.action[:], 'b-', label='Certified Inputs')
    ax_act.plot(mpsc_results.uncertified_action[:], 'r--', label='Uncertified Input')
    ax_act.legend()
    ax_act.set_title('Input comparison')
    ax_act.set_xlabel('Step')
    ax_act.set_ylabel('Input')
    ax_act.set_box_aspect(0.5)

    _, ax = plt.subplots()
    ax.plot(certified_results.obs[:,2], certified_results.obs[:,3],'.-', label='Certified')
    ax.plot(certified_results.obs[certified_results.corrections>1e-6, 2], certified_results.obs[certified_results.corrections>1e-6, 3], 'r.', label='Modified')
    uncert_end = results.obs.shape[0]
    ax.plot(results.obs[:uncert_end, 2], results.obs[:uncert_end, 3], 'r--', label='Uncertified')
    ax.axvline(x=-0.2, color='r', label='Limit')
    ax.axvline(x=0.2, color='r')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    ax.set_box_aspect(0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
