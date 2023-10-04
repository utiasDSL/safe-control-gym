'''Template training/plotting/testing script.'''

import os
from functools import partial

import torch

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import (mkdirs, set_device_from_config,
                                          set_dir_from_config,
                                          set_seed_from_config)


def train(config):
    '''Training template.

    Usage:
        * to restore from a previous training, additionally use `--restore {dir_path}`
            where `dir_path` is the output folder from previous training.
    '''
    # Experiment setup.
    if not config.restore:
        set_dir_from_config(config)

    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config
                       )

    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()

    if config.restore:
        control_agent.load(os.path.join(config.restore, 'model_latest.pt'))

    # Training.
    control_agent.learn()
    control_agent.close()
    print('Training done.')


def make_plots(config):
    '''Produces plots for logged stats during training.
    Usage
        * use with `--func plot` and `--restore {dir_path}` where `dir_path` is
            the experiment folder containing the logs.
        * save figures under `dir_path/plots/`.
    '''
    # Define source and target log locations.
    log_dir = os.path.join(config.output_dir, 'logs')
    plot_dir = os.path.join(config.output_dir, 'plots')
    mkdirs(plot_dir)
    plot_from_logs(log_dir, plot_dir, window=3)
    print('Plotting done.')


MAIN_FUNCS = {'train': train, 'plot': make_plots}


if __name__ == '__main__':
    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--thread', type=int, default=0, help='number of threads to use (set by torch).')
    config_dict = fac.merge()

    # System settings.
    if config_dict.thread > 0:
        # E.g. set single thread for less context switching
        torch.set_num_threads(config_dict.thread)

    func = MAIN_FUNCS.get(config_dict.func, None)
    if func is None:
        raise Exception(f'Main function {config_dict.func} not supported.')
    func(config_dict)
