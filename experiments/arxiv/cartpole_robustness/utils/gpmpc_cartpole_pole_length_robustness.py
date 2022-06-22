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
import shelve

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config

# To set relative pathing of experiment imports.
import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__, "../../../utils/")))
from gpmpc_plotting_utils import plot_robustness_rmse, plot_robustness, plot_all_robustness_runs, table_csv, plot_robustness_from_csv

def main(config):
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    config.algo_config.output_dir = config.output_dir
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )
    ctrl.reset()
    #_ = ctrl.learn()
    data_path = os.path.join(config.algo_config.gp_model_path, 'data.npz')
    data = np.load(data_path)
    data_inputs = data['data_inputs']
    data_targets = data['data_targets']
    _ = ctrl.learn(input_data=data_inputs,
                   target_data=data_targets)

    num_eval_episodes = config.num_eval_episodes
    coeff = np.array(config.coeff)
    cart_masses = 1.0*coeff
    pole_masses = 0.1*coeff
    pole_lengths = 0.5*coeff
    parameter_tests = {'cart_mass': cart_masses,
                       'pole_mass': pole_masses,
                       'pole_length': pole_lengths}

    test_results = {}

    for i in range(coeff.shape[0]):
        inertial_prop = {}
        seed = config.seed
        for parameter in config.parameters_to_vary:
            inertial_prop[parameter] = parameter_tests[parameter][i]
        test_results[i] = {}
        for episode in range(num_eval_episodes):
            print("EPOCH: %s" % i)
            print("EPIOSDE: %s" % episode)
            test_env = env_func(inertial_prop=inertial_prop, seed=seed)
            run_results = ctrl.run(env=test_env,
                                   terminate_run_on_done=config.terminate_test_on_done)
            test_results[i][episode] = run_results
            seed = config.seed + episode + 1
            test_env.close()

        np.savez(os.path.join(config.output_dir,
                              'data.npz'),
                              test_results=test_results,
                              coeff=parameter_tests['pole_length'],
                              ctrl_freq=config.task_config.ctrl_freq)

    label = 'Pole Length'
    plot_robustness(test_results, parameter_tests['pole_length'], label, config.output_dir)
    plot_robustness_rmse(test_results, parameter_tests['pole_length'], label, config.output_dir)
    plot_all_robustness_runs(test_results, parameter_tests['pole_length'], config.output_dir)
    table_csv(test_results, config.output_dir)
    fname = os.path.join(config.output_dir, 'rmse_robust_plot.csv')
    plot_robustness_from_csv(fname, 'Cartpole Pole Length Robustness', 'l (m)')
    return test_results




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
        fname =config.plot_dir
        plot_robustness_from_csv(fname, 'Cartpole Pole Length Robustness', 'l (m)')
