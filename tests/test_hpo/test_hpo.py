import os
import sys

import pytest

from examples.hpo.hpo_experiment import hpo
from safe_control_gym.utils.configuration import ConfigFactory


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ilqr', 'gp_mpc', 'ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
@pytest.mark.parametrize('SAMPLER', ['optuna', 'vizier'])
def test_hpo_cartpole(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, SAMPLER):
    '''Test HPO for one single trial.
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./examples/hpo/results/{ALGO}'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # delete database
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db-journal'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db-journal')
    if os.path.exists(f'{ALGO}_hpo_endpoint.yaml'):
        os.system(f'rm {ALGO}_hpo_endpoint.yaml')

    if ALGO == 'ilqr':
        PRIOR = '100'
    elif ALGO == 'gp_mpc' or ALGO == 'gpmpc_acados':
        PRIOR = '200'

    # check if the config file exists
    TASK_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    HPO_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{ALGO}_{SYS}_hpo.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
    assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            pytest.skip('SAFETY_FILTER is only supported for ilqr')
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./examples/hpo/{SYS}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST = 'one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='optuna', help='which sampler to use in HPO.')
    config = fac.merge()
    config.hpo_config.trials = 1
    config.hpo_config.repetitions = 1
    config.sampler = SAMPLER

    # hyperparameter configurations (to make tests faster)
    # should belong to the distributions defined in the search space
    if 'optimization_iterations' in config.hpo_config.hps_config:
        d = len(config.hpo_config.hps_config.optimization_iterations)
        config.hpo_config.hps_config.optimization_iterations = [2400] * d
    if 'num_epochs' in config.hpo_config.hps_config:
        config.hpo_config.hps_config.num_epochs = 2
    if 'max_env_steps' in config.hpo_config.hps_config:
        config.hpo_config.hps_config.max_env_steps = 30000

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # delete database
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db')
    if os.path.exists(f'{ALGO}_hpo_{SAMPLER}.db-journal'):
        os.system(f'rm {ALGO}_hpo_{SAMPLER}.db-journal')
    if os.path.exists(f'{ALGO}_hpo_endpoint.yaml'):
        os.system(f'rm {ALGO}_hpo_endpoint.yaml')
