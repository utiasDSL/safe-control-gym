"""This script runs the MPSC experiment in our Annual Reviews article.

See Figure 8 in https://arxiv.org/pdf/2108.06266.pdf.

"""
import os
import sys
import shutil
import matplotlib.pyplot as plt
from munch import munchify
from functools import partial

from safe_control_gym.utils.utils import read_file
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory


def main():
    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    # Create controller from PPO YAML.
    ppo_config_dir = os.path.dirname(os.path.abspath(__file__))+'/config_overrides'
    ppo_dict = read_file(os.path.join(ppo_config_dir,'unsafe_ppo_config.yaml'))
    ppo_config = munchify(ppo_dict)
    # Setup PPO controller.
    ppo_ctrl = make(ppo_config.algo,
                    env_func,
                    **ppo_config.algo_config)
    # Load state_dict from trained PPO.
    ppo_model_dir = os.path.dirname(os.path.abspath(__file__))+'/unsafe_ppo_model'
    ppo_ctrl.load(os.path.join(ppo_model_dir,'unsafe_ppo_model_30000.pt'))  # Show violation.
    # Remove temporary files and directories
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp')
    # Setup MPSC.
    ctrl = make(config.algo,
                env_func,
                rl_controller=ppo_ctrl,
                **config.algo_config)
    ctrl.reset()
    ctrl.learn()
    test_env = env_func()
    uncertified_env = env_func()
    results = ctrl.run(env=test_env,
                       uncertified_env=uncertified_env)
    ctrl.close()
    fig_obs, ax_obs = plt.subplots()
    ax_obs.plot(results.obs[:, 0], results.obs[:, 2], '.-', label='Certified')
    ax_obs.plot(results.uncertified_obs[:10, 0], results.uncertified_obs[:10, 2], 'r--', label='Uncertified')
    ax_obs.plot(results.obs[results.corrections>1e-6, 0], results.obs[results.corrections>1e-6, 2], 'r.', label='Modified')
    ax_obs.legend()
    ax_obs.set_title('State Space')
    ax_obs.set_xlabel(r'$x$')
    ax_obs.set_ylabel(r'$\theta$')
    ax_obs.set_box_aspect(0.5)
    fig_act, ax_act = plt.subplots()
    ax_act.plot(results.actions[:], 'b-', label='Certified Inputs')
    ax_act.plot(results.learning_actions[:], 'r--', label='Uncertified Input')
    ax_act.legend()
    ax_act.set_title('Input comparison')
    ax_act.set_xlabel('Step')
    ax_act.set_ylabel('Input')
    ax_act.set_box_aspect(0.5)
    fig, ax = plt.subplots()
    ax.plot(results.obs[:,2], results.obs[:,3],'.-', label='Certified')
    modified_inds = results.corrections>1e-6
    ax.plot(results.obs[results.corrections>1e-6, 2], results.obs[results.corrections>1e-6, 3], 'r.', label='Modified')
    uncert_end = results.uncertified_obs.shape[0]
    ax.plot(results.uncertified_obs[:uncert_end, 2], results.uncertified_obs[:uncert_end, 3], 'r--', label='Uncertified')
    ax.axvline(x=-0.2, color='r', label='Limit')
    ax.axvline(x=0.2, color='r')
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_box_aspect(0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
