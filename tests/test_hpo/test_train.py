import os
import sys

import munch
import pytest

from examples.hpo.rl.rl_experiment import train
from safe_control_gym.hyperparameters.database import create, drop
from safe_control_gym.utils.configuration import ConfigFactory


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ppo', 'sac'])
@pytest.mark.parametrize('PRIOR', [''])
def test_train(SYS, TASK, ALGO, PRIOR):
    '''Test training given a set of hyperparameters.
    '''

    # output_dir
    output_dir = f'./examples/hpo/rl/{ALGO}/results'
    # delete output_dir if exists
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # optimized hp path
    if ALGO == 'ppo':
        opt_hp_path = 'examples/hpo/rl/ppo/config_overrides/cartpole/optimized_hyperparameters.yaml'
    elif ALGO == 'sac':
        opt_hp_path = 'examples/hpo/rl/sac/config_overrides/cartpole/optimized_hyperparameters.yaml'

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
    config = fac.merge()

    train(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
