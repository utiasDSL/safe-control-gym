import os
import sys

import munch
import pytest

from examples.hpo.hpo_experiment import train, hpo
from safe_control_gym.hyperparameters.database import create, drop
from safe_control_gym.utils.configuration import ConfigFactory


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ilqr', 'gp_mpc', 'gpmpc_acados','ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
@pytest.mark.parametrize('HYPERPARAMETER', ['default'])
def test_train_cartpole(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, HYPERPARAMETER):
    '''Test training rl/lbc given a set of hyperparameters.
    '''

    # output_dir
    output_dir = f'./benchmarking_sim/results/{ALGO}'
    # delete output_dir if exists
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # optimized hp path
    if ALGO == 'ilqr' or ALGO == 'linear_mpsc':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        else:
            raise ValueError('HYPERPARAMETER must be default')
        PRIOR = '100'
    elif ALGO == 'gp_mpc' or ALGO == 'gpmpc_acados':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        else:
            raise ValueError('HYPERPARAMETER must be default')
        PRIOR = '200'
    
    # check if the config file exists
    TASK_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST='one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--safety_filter', SAFETY_FILTER,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', output_dir,
                            ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', output_dir,
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

@pytest.mark.parametrize('SYS', ['quadrotor_2D_attitude'])
@pytest.mark.parametrize('TASK', ['tracking'])
@pytest.mark.parametrize('ALGO', ['ilqr', 'gp_mpc', 'gpmpc_acados','ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
@pytest.mark.parametrize('HYPERPARAMETER', ['default', 'optimized'])
def test_train_quadrotor(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, HYPERPARAMETER):
    '''Test training rl/lbc given a set of hyperparameters.
    '''

    # output_dir
    output_dir = f'./benchmarking_sim/results/{ALGO}'
    # delete output_dir if exists
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    SYS_NAME = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS

    # optimized hp path
    if ALGO == 'ilqr' or ALGO == 'linear_mpsc':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        elif HYPERPARAMETER == 'optimized':
            opt_hp_path = './benchmarking_sim/quadrotor/config_overrides/ilqr_optimized_hyperparameters.yaml'
        PRIOR = '100'
    elif ALGO == 'gp_mpc' or ALGO == 'gpmpc_acados':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        elif HYPERPARAMETER == 'optimized':
            opt_hp_path = './benchmarking_sim/quadrotor/config_overrides/gpmpc_acados_optimized_hyperparameters.yaml'
        PRIOR = '200'
    elif ALGO == 'ppo':
        if HYPERPARAMETER == 'default':
            opt_hp_path = ''
        else:
            opt_hp_path = './benchmarking_sim/quadrotor/config_overrides/ppo_optimized_hyperparameters.yaml'
    
    # check if the config file exists
    TASK_CONFIG_PATH = f'./benchmarking_sim/{SYS_NAME}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./benchmarking_sim/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST='one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--safety_filter', SAFETY_FILTER,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--opt_hps', opt_hp_path,
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', output_dir,
                            ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                        '--opt_hps', opt_hp_path,
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', output_dir,
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

@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK', ['stab'])
@pytest.mark.parametrize('ALGO', ['ilqr', 'gp_mpc', 'gpmpc_acados', 'ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
@pytest.mark.parametrize('SAMPLER', ['Optuna', 'Vizier'])
def test_hpo_cartpole(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, SAMPLER):
    '''Test HPO for one single trial using MySQL database.
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./benchmarking_sim/results/{ALGO}'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    if ALGO == 'ilqr':
        PRIOR = '100'
    elif ALGO == 'gp_mpc' or ALGO == 'gpmpc_acados':
        PRIOR = '200'

    # check if the config file exists
    TASK_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    HPO_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{ALGO}_{SYS}_hpo.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
    assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST='one_step_cost'
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
    fac.add_argument('--sampler', type=str, default='Optuna', help='which sampler to use in HPO.')
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



@pytest.mark.parametrize('SYS', ['quadrotor_2D_attitude'])
@pytest.mark.parametrize('TASK', ['tracking'])
@pytest.mark.parametrize('ALGO', ['ilqr', 'gp_mpc', 'gpmpc_acados', 'ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
@pytest.mark.parametrize('SAMPLER', ['Optuna', 'Vizier'])
def test_hpo_quadrotor(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER, SAMPLER):
    '''Test HPO for one single trial using MySQL database.
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./benchmarking_sim/results/{ALGO}'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # # drop the database if exists
    # drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # # create database
    # create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # delete .db
    if os.path.exists(f'{ALGO}_hpo.db'):
        os.system(f'rm -rf {ALGO}_hpo.db')
    # delete .db-journal
    if os.path.exists(f'{ALGO}_hpo.db-journal'):
        os.system(f'rm -rf {ALGO}_hpo.db-journal')

    SYS_NAME = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS

    if ALGO == 'ilqr':
        PRIOR = '100'
    elif ALGO == 'gp_mpc' or ALGO == 'gpmpc_acados':
        PRIOR = '200'

    # check if the config file exists
    TASK_CONFIG_PATH = f'./benchmarking_sim/{SYS_NAME}/config_overrides/{SYS}_{TASK}.yaml'
    ALGO_CONFIG_PATH = f'./benchmarking_sim/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    HPO_CONFIG_PATH = f'./benchmarking_sim/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_hpo.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
    assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./benchmarking_sim/{SYS}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST='one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
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
                        '--task', SYS_NAME,
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
    fac.add_argument('--sampler', type=str, default='Optuna', help='which sampler to use in HPO.')
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
    # drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
