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

import munch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config

# To set relative pathing of experiment imports.
import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__, "../../../utils/")))
from gpmpc_plotting_utils import filter_sequences, gather_training_samples, make_quad_plots, make_traking_plot, plot_data_eff_from_csv


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

    num_epochs = config.num_epochs
    num_train_episodes_per_epoch = config.num_train_episodes_per_epoch
    num_test_episodes_per_epoch = config.num_test_episodes_per_epoch
    num_samples = config.num_samples
    train_runs = {0: {}}
    test_runs = {0: {}}
    train_envs = []
    for epoch in range(num_epochs):
        train_envs.append(env_func(randomized_init=False))
    test_envs = []
    for epoch in range(num_epochs):
        test_envs.append(env_func(randomized_init=False))

    for episode in range(num_train_episodes_per_epoch):
        run_results = ctrl.prior_ctrl.run(env=train_envs[0],
                                          terminate_run_on_done=config.terminate_train_on_done)
        train_runs[0].update({episode: munch.munchify(run_results)})
        ctrl.reset()
    for test_ep in range(num_test_episodes_per_epoch):
        run_results = ctrl.run(env=test_envs[0],
                               terminate_run_on_done=config.terminate_test_on_done)
        test_runs[0].update({test_ep: munch.munchify(run_results)})
    ctrl.reset()

    for epoch in range(1, num_epochs):
        # only take data from the last episode from the last epoch
        x_seq, actions, x_next_seq = gather_training_samples(train_runs, epoch-1, num_samples)
        #x_seq, actions, x_next_seq = gather_training_samples_from_all_data(train_runs, num_samples*epoch)
        x_seq_filtered, actions_filtered, x_next_seq_filtered = filter_sequences(x_seq, actions, x_next_seq, 1)

        train_inputs, train_outputs = ctrl.preprocess_training_data(x_seq_filtered,
                                                                    actions_filtered,
                                                                    x_next_seq_filtered)
        #train_inputs, train_outputs = ctrl.preprocess_training_data(x_seq, actions, x_next_seq)
        _ = ctrl.learn(input_data=train_inputs, target_data=train_outputs)

        # Test new policy.
        test_runs[epoch] = {}
        for test_ep in range(num_test_episodes_per_epoch):
            ctrl.reset()
            ctrl.x_prev = test_runs[epoch-1][0]['obs'][:ctrl.T+1,:].T
            ctrl.u_prev = test_runs[epoch-1][0]['action'][:ctrl.T,:].T
            run_results = ctrl.run(env=test_envs[epoch],
                                   terminate_run_on_done=config.terminate_test_on_done)
            test_runs[epoch].update({test_ep: munch.munchify(run_results)})
        # gather training data
        train_runs[epoch] = {}
        for episode in range(num_train_episodes_per_epoch):
            ctrl.reset()
            ctrl.x_prev = test_runs[epoch-1][0]['obs'][:ctrl.T+1,:].T
            ctrl.u_prev = test_runs[epoch-1][0]['action'][:ctrl.T,:].T
            run_results = ctrl.run(env=train_envs[epoch],
                                   terminate_run_on_done=config.terminate_train_on_done)
            train_runs[epoch].update({episode: munch.munchify(run_results)})

        trajectory = test_envs[0].X_GOAL
        make_traking_plot(test_runs, trajectory, config.output_dir, impossible=False)


    np.savez(os.path.join(config.output_dir, 'data'),
             train_runs=train_runs,
             test_runs=test_runs,
             num_epochs=num_epochs,
             num_train_episodes_per_epoch=num_train_episodes_per_epoch,
             num_test_episodes_per_epoch=num_test_episodes_per_epoch,
             num_samples=num_samples,
             trajectory=trajectory,
             ctrl_freq=config.task_config.ctrl_freq,
             data_inputs=ctrl.data_inputs,
             data_targets=ctrl.data_targets)
    make_quad_plots(test_runs, train_runs, trajectory, test_envs[0].state_dim, config.output_dir)
    plt.close('all')
    make_traking_plot(test_runs, trajectory, config.output_dir, impossible=False)
    fname = os.path.join(config.output_dir, 'figs', 'rmse_xz_error_learning_curve.csv')
    plt.figure()
    plot_data_eff_from_csv(fname,
                           'Quadrotor Data Efficiency')
    return train_runs, test_runs

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
        train_runs, test_runs = main(config)
    else:
        fname = config.plot_dir
        plot_data_eff_from_csv(fname,
                               'Quadrotor Data Efficiency')
