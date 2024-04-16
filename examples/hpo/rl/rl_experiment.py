"""Template training/plotting/testing script.

"""
import os
import pickle
import sys
from functools import partial

import munch
import torch
import yaml

from safe_control_gym.hyperparameters.hpo import HPO
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import (mkdirs, save_video, set_device_from_config, set_dir_from_config,
                                          set_seed_from_config)


def hpo(config):
    """Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.

    """

    # Experiment setup.
    if not config.restore and config.hpo_config.hpo:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # Put config.seed into config.hpo_config.seed
    config.hpo_config.seed = config.seed
    # HPO
    hpo = HPO(config.algo,
              config.task,
              config.sampler,
              config.load_study,
              config.output_dir,
              config.task_config,
              config.hpo_config,
              **config.algo_config)
    if config.hpo_config.hpo:
        hpo.hyperparameter_optimization()
        print('Hyperparameter optimization done.')
    elif config.hpo_config.perturb_hps and config.opt_hps != '':
        hpo._perturb_hps(hp_path=config.opt_hps)
        print('Perturbed hyperparameter files have been produced.')
    else:
        raise ValueError('hpo or perturb_hps must be set to True.')


def train(config):
    """Training template.

    Usage:
        * to start training, use with `--func train`.
        * to restore from a previous training, additionally use `--restore {dir_path}`
            where `dir_path` is the output folder from previous training.

    """
    # Override algo_config with given yaml file
    if config.opt_hps == '':
        # if no opt_hps file is given
        pass
    else:
        # if opt_hps file is given
        with open(config.opt_hps, 'r') as f:
            opt_hps = yaml.load(f, Loader=yaml.FullLoader)
        for hp in opt_hps:
            config.algo_config[hp] = opt_hps[hp]
    # Experiment setup.
    if not config.restore:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)
    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    # Note:
    # eval_env will take config.seed * 111 as its seed
    # env will take config.seed as its seed
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


MAIN_FUNCS = {'hpo': hpo, 'train': train}


if __name__ == '__main__':

    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception('Main function {} not supported.'.format(config.func))
    func(config)
