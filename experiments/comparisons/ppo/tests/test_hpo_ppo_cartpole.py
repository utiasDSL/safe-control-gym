import sys, os
import pytest
import munch
import time


from safe_control_gym.utils.configuration import ConfigFactory
from experiments.comparisons.ppo.ppo_experiment import hpo, train
from safe_control_gym.hyperparameters.database import create, drop


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo'])
@pytest.mark.parametrize('PRIOR',['', '150'])
def test_hpo_ppo_cartpole(SYS, TASK, ALGO, PRIOR):
    '''Test HPO for ppo on cartpole stab task for one single trial
        (create a study from scratch)
    '''

    # output_dir
    output_dir = './experiments/comparisons/ppo/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': 'ppo_hpo'}))
    # create database
    create(munch.Munch({'tag': 'ppo_hpo'}))

    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '7',
                    '--use_gpu', 'True'
                    ]

    fac = ConfigFactory()
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    config = fac.merge()
    config.hpo_config.trials = 1

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': 'ppo_hpo'}))

@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo'])
@pytest.mark.parametrize('STRATEGY',['1', '2', '3', '4'])
def test_hpo_stategy_ppo_cartpole(SYS, TASK, ALGO, STRATEGY):
    '''Test HPO strategies for ppo on cartpole stab task for one single trial
        (create a study from scratch)
    '''

    # output_dir
    output_dir = './experiments/comparisons/ppo/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': 'ppo_hpo'}))
    # create database
    create(munch.Munch({'tag': 'ppo_hpo'}))

    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_.yaml',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{STRATEGY}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '7',
                    '--use_gpu', 'True'
                    ]

    fac = ConfigFactory()
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    config = fac.merge()
    config.hpo_config.trials = 1

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': 'ppo_hpo'}))

@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo'])
@pytest.mark.parametrize('PRIOR',['', '150'])
@pytest.mark.parametrize('LOAD', [False, True])
def test_hpo_ppo_cartpole_parallelism(SYS, TASK, ALGO, PRIOR, LOAD):
    '''Test HPO for ppo on cartpole stab task in parallel.'''

    # if LOAD is False, create a study from scratch
    if not LOAD:
        # drop the database if exists
        drop(munch.Munch({'tag': 'ppo_hpo'}))
        # create database
        create(munch.Munch({'tag': 'ppo_hpo'}))
        # output_dir
        output_dir = './experiments/comparisons/ppo/results'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./experiments/comparisons/ppo/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

        fac = ConfigFactory()
        fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
        config = fac.merge()
        config.hpo_config.trials = 1

        hpo(config)
    # if LOAD is True, load a study from a previous HPO study
    else:
        # first, wait a bit untill the HPO study is created
        time.sleep(3)
        # output_dir
        output_dir = './experiments/comparisons/ppo/results'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./experiments/comparisons/ppo/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--seed', '8',
                        '--use_gpu', 'True'
                        ]

        fac = ConfigFactory()
        fac.add_argument("--load_study", type=bool, default=True, help="whether to load study from a previous HPO.")
        config = fac.merge()
        config.hpo_config.trials = 1

        hpo(config)

        # delete output_dir
        if os.path.exists(output_dir):
            os.system(f'rm -rf {output_dir}')

        # drop database
        drop(munch.Munch({'tag': 'ppo_hpo'}))


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo'])
@pytest.mark.parametrize('PRIOR',['', '150'])
def test_train_ppo_cartpole(SYS, TASK, ALGO, PRIOR):
    '''Test training for ppo on cartpole stab task given a set of hyperparameters.
    '''
    
    # output_dir
    output_dir = './experiments/comparisons/ppo/results'
    # delete output_dir if exists
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': 'ppo_hpo'}))
    # create database
    create(munch.Munch({'tag': 'ppo_hpo'}))

    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/ppo/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '6',
                    '--use_gpu', 'True'
                    ]
    
    fac = ConfigFactory()
    fac.add_argument("--opt_hps", type=str, default="", help="yaml file as a result of HPO.")
    config = fac.merge()

    train(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': 'ppo_hpo'}))