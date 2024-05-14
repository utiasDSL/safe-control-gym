import os
import sys
import time

import munch
import pytest

from examples.hpo.hpo_experiment import hpo
from safe_control_gym.hyperparameters.database import create, drop
from safe_control_gym.utils.configuration import ConfigFactory


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ppo', 'sac', 'gp_mpc'])
@pytest.mark.parametrize('SAMPLER', ['TPESampler', 'RandomSampler'])
def test_hpo(SYS, TASK, ALGO, SAMPLER):
    '''Test HPO for one single trial using MySQL database.
        (create a study from scratch)
    '''
    pytest.skip('Takes too long.')

    # output_dir
    output_dir = './examples/hpo/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    if ALGO == 'gp_mpc':
        PRIOR = '150'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                        '--output_dir', output_dir,
                        '--seed', '1',
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
    config = fac.merge()
    config.hpo_config.trials = 1
    config.hpo_config.repetitions = 1
    config.hpo_config.use_database = True
    config.sampler = SAMPLER

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ppo', 'sac', 'gp_mpc'])
@pytest.mark.parametrize('LOAD', [False, True])
@pytest.mark.parametrize('SAMPLER', ['TPESampler', 'RandomSampler'])
def test_hpo_parallelism(SYS, TASK, ALGO, LOAD, SAMPLER):
    '''Test HPO for in parallel.'''

    pytest.skip('Takes too long.')

    # if LOAD is False, create a study from scratch
    if not LOAD:
        # drop the database if exists
        drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
        # create database
        create(munch.Munch({'tag': f'{ALGO}_hpo'}))
        # output_dir
        output_dir = './examples/hpo/results'

        if ALGO == 'gp_mpc':
            PRIOR = '150'
            sys.argv[1:] = ['--algo', ALGO,
                            '--task', SYS,
                            '--overrides',
                                f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                                f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                                f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                            '--output_dir', output_dir,
                            '--seed', '1',
                            ]
        else:
            sys.argv[1:] = ['--algo', ALGO,
                            '--task', SYS,
                            '--overrides',
                                f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                                f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
                                f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                            '--output_dir', output_dir,
                            '--seed', '7',
                            '--use_gpu', 'True'
                            ]

        fac = ConfigFactory()
        fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
        fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
        config = fac.merge()
        config.hpo_config.trials = 1
        config.hpo_config.repetitions = 1
        config.hpo_config.use_database = True
        config.sampler = SAMPLER

        hpo(config)
    # if LOAD is True, load a study from a previous HPO study
    else:
        # first, wait a bit untill the HPO study is created
        time.sleep(3)
        # output_dir
        output_dir = './examples/hpo/results'
        if ALGO == 'gp_mpc':
            PRIOR = '150'
            sys.argv[1:] = ['--algo', ALGO,
                            '--task', SYS,
                            '--overrides',
                                f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                                f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                                f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                            '--output_dir', output_dir,
                            '--seed', '1',
                            ]
        else:
            sys.argv[1:] = ['--algo', ALGO,
                            '--task', SYS,
                            '--overrides',
                                f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                                f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
                                f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                            '--output_dir', output_dir,
                            '--seed', '8',
                            '--use_gpu', 'True'
                            ]

        fac = ConfigFactory()
        fac.add_argument('--load_study', type=bool, default=True, help='whether to load study from a previous HPO.')
        fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
        config = fac.merge()
        config.hpo_config.trials = 1
        config.hpo_config.repetitions = 1
        config.sampler = SAMPLER

        hpo(config)

        # delete output_dir
        if os.path.exists(output_dir):
            os.system(f'rm -rf {output_dir}')

        # drop database
        drop(munch.Munch({'tag': f'{ALGO}_hpo'}))


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ppo', 'sac', 'gp_mpc'])
@pytest.mark.parametrize('SAMPLER', ['TPESampler', 'RandomSampler'])
def test_hpo_without_database(SYS, TASK, ALGO, SAMPLER):
    '''Test HPO for one single trial without using MySQL database.
        (create a study from scratch)
    '''
    pytest.skip('Takes too long.')

    # output_dir
    output_dir = './examples/hpo/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    if ALGO == 'gp_mpc':
        PRIOR = '150'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./examples/hpo/gp_mpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                        '--output_dir', output_dir,
                        '--seed', '1',
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
    config = fac.merge()
    config.hpo_config.trials = 1
    config.hpo_config.repetitions = 1
    config.hpo_config.use_database = False
    config.sampler = SAMPLER

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
