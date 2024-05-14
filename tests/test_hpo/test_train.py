import os
import sys

import munch
import pytest

from examples.hpo.hpo_experiment import train
from safe_control_gym.hyperparameters.database import create, drop
from safe_control_gym.utils.configuration import ConfigFactory


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ppo', 'sac', 'gp_mpc'])
@pytest.mark.parametrize('HYPERPARAMETER', ['default', 'optimimum'])
def test_train(SYS, TASK, ALGO, HYPERPARAMETER):
    '''Test training rl/lbc given a set of hyperparameters.'''
    pytest.skip('Takes too long.')

    # output_dir
    output_dir = './examples/hpo/results'
    # delete output_dir if exists
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # optimized hp path
    if ALGO == 'ppo':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        elif HYPERPARAMETER == 'optimimum':
            opt_hp_path = 'examples/hpo/rl/ppo/config_overrides/cartpole/optimized_hyperparameters.yaml'
        else:
            raise ValueError('HYPERPARAMETER must be either default or optimimum')
    elif ALGO == 'sac':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        elif HYPERPARAMETER == 'optimimum':
            opt_hp_path = 'examples/hpo/rl/sac/config_overrides/cartpole/optimized_hyperparameters.yaml'
        else:
            raise ValueError('HYPERPARAMETER must be either default or optimimum')
    elif ALGO == 'gp_mpc':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        elif HYPERPARAMETER == 'optimimum':
            opt_hp_path = 'examples/hpo/gp_mpc/config_overrides/cartpole/optimized_hyperparameters.yaml'
        else:
            raise ValueError('HYPERPARAMETER must be either default or optimimum')
        PRIOR = '150'

    if ALGO == 'gp_mpc':
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/{ALGO}/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--opt_hps', opt_hp_path,
                        '--seed', '2'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
                        '--output_dir', output_dir,
                        '--tag', 's1',
                        '--opt_hps', opt_hp_path,
                        '--seed', '6',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    config = fac.merge()

    train(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
