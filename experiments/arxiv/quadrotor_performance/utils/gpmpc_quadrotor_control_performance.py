"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

"""
import os
import matplotlib.pyplot as plt
import munch
import yaml
import numpy as np
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config

# To set relative pathing of experiment imports.
import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__, "../../../utils/")))
from gpmpc_plotting_utils import plot_ctrl_perf

def trajectory_plot_csv(traj_data, action, dir, ref=None):
    times = np.arange(traj_data.shape[0])
    action = np.atleast_2d(action)

    fig, ax = plt.subplots(traj_data.shape[1] + action.shape[1], 1, sharex='col')
    for i in range(traj_data.shape[1]):
        ax[i].plot(traj_data[:,i])
        if ref is not None:
            ax[i].plot(ref[:,i], '--r', label='ref')
        ax[i].set_ylabel('x%s' % i)
        ax[i].set_xlabel('Time (s)')
    i += 1
    for j in range(action.shape[1]):
        ax[i+j].plot(action[:,j])

    action = np.vstack((action, np.zeros((1,action.shape[1]))))

    plt.savefig(os.path.join(dir,'trajectory_plot.png'))

    data = np.hstack((times[:, None], traj_data, action))
    fname = os.path.join(dir, 'trajectory_plot.csv')
    header = 'Time step,x position, x velocity, theta position, theta velocity,action'
    np.savetxt(fname, data, delimiter=',', header=header)


def main(config):
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    config.algo_config.output_dir = config.output_dir
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )
    ctrl.reset()

    data_path = os.path.join(config.algo_config.gp_model_path, 'data.npz')
    data = np.load(data_path)
    data_inputs = data['data_inputs']
    data_targets = data['data_targets']
    _ = ctrl.learn(input_data=data_inputs,
                   target_data=data_targets)

    test_results = {}
    init_state = {'init_x': 0.0,
                  'init_x_dot': 0.0,
                  'init_z': 0.0,
                  'init_x_dot': 0.0,
                  'init_theta': 0.0,
                  'init_theta_dot': 0.0}



    test_env = env_func(init_state=init_state,
                        randomized_init=False,
                        seed=config.seed)
    test_env.action_space.seed(config.seed)
    run_results = ctrl.run(env=test_env)

    np.savez(os.path.join(config.output_dir,
                          'data.npz'),
                          run_results=run_results,
                          init_state=init_state,
                          ctrl_freq=config.task_config.ctrl_freq)


    trajectory_plot_csv(run_results['obs'],
                        run_results['action'],
                        config.output_dir,
                        ref=test_env.X_GOAL)
    plt.show()
    #plt.plot(run_results['obs'][:,0], run_results['obs'][:,2], 'b')
    #plt.plot(test_env.X_GOAL[:,0], test_env.X_GOAL[:,2], 'r')

    return run_results




if __name__ == "__main__":
    fac = ConfigFactory()
    fac.add_argument("--plot_dir", type=str, default='', help="Create plot from CSV file.")
    config = fac.merge()
    set_dir_from_config(config)
    mkdirs(config.output_dir)

    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)

    if config.plot_dir == '':
        test_runs = main(config)
    else:
        fname = config.plot_dir
        plot_ctrl_perf(fname)
