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
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('HYPERPARAMETER', ['default', 'optimimum'])
def test_train_cartpole(SYS, TASK, ALGO, PRIOR, HYPERPARAMETER):
    '''Test training rl/lbc given a set of hyperparameters.
    '''

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
                        '--seed', '2',
                        '--use_gpu', 'True'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--tag', 's1',
                        '--opt_hps', opt_hp_path,
                        '--seed', '6',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    fac.add_argument('--plot_best', type=bool, default=False, help='plot best agent trajectory.')
    config = fac.merge()

    train(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))

@pytest.mark.parametrize('SYS', ['quadrotor_2D', 'quadrotor_2D_attitude'])
@pytest.mark.parametrize('TASK', ['track'])
@pytest.mark.parametrize('ALGO', ['ppo', 'sac', 'gp_mpc'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('HYPERPARAMETER', ['default', 'optimimum'])
def test_train_quad(SYS, TASK, ALGO, PRIOR, HYPERPARAMETER):
    '''Test training rl/lbc given a set of hyperparameters.
    '''
    SYS_NAME = 'quadrotor' if 'quadrotor' in SYS else SYS
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
    if ALGO == 'ppo' or ALGO == 'sac' or ALGO == 'gp_mpc':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        elif HYPERPARAMETER == 'optimimum':
            raise ValueError('optimimum hyperparameters are not available for quadrotor')
        else:
            raise ValueError('HYPERPARAMETER must be either default or optimimum')
        
    if ALGO == 'gp_mpc':
        PRIOR = '150'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            f'./examples/hpo/{ALGO}/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--opt_hps', opt_hp_path,
                        '--seed', '2',
                        '--use_gpu', 'True'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                            '--task', SYS_NAME,
                            '--overrides',
                                f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                                f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            '--output_dir', output_dir,
                            '--tag', 's1',
                            '--opt_hps', opt_hp_path,
                            '--seed', '6',
                            '--use_gpu', 'True'
                            ]

    fac = ConfigFactory()
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    fac.add_argument('--plot_best', type=bool, default=False, help='plot best agent trajectory.')
    config = fac.merge()

    train(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
