import sys, os
import pytest
import munch
import time


from safe_control_gym.utils.configuration import ConfigFactory
from experiments.comparisons.rl.rl_experiment import hpo, train
from safe_control_gym.hyperparameters.database import create, drop


@pytest.mark.parametrize('SYS', ['cartpole', 'quadrotor'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
@pytest.mark.parametrize('PRIOR',[''])
@pytest.mark.parametrize('SAMPLER',['TPESampler', 'RandomSampler'])
def test_hpo(SYS, TASK, ALGO, PRIOR, SAMPLER):
    '''Test HPO for one single trial
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./experiments/comparisons/rl/{ALGO}/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '7',
                    '--use_gpu', 'True'
                    ]

    fac = ConfigFactory()
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    fac.add_argument("--sampler", type=str, default="TPESampler", help="which sampler to use in HPO.")
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
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
@pytest.mark.parametrize('STRATEGY',['1', '2', '3', '4', '5'])
@pytest.mark.parametrize('SAMPLER',['TPESampler', 'RandomSampler'])
def test_hpo_stategy(SYS, TASK, ALGO, STRATEGY, SAMPLER):
    '''Test HPO strategies for one single trial
        (create a study from scratch)
    '''

    # output_dir
    output_dir = f'./experiments/comparisons/rl/{ALGO}/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_.yaml',
                        f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{STRATEGY}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '7',
                    '--use_gpu', 'True'
                    ]

    fac = ConfigFactory()
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    fac.add_argument("--sampler", type=str, default="TPESampler", help="which sampler to use in HPO.")
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
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
@pytest.mark.parametrize('PRIOR',[''])
@pytest.mark.parametrize('LOAD', [False, True])
@pytest.mark.parametrize('SAMPLER',['TPESampler', 'RandomSampler'])
def test_hpo_parallelism(SYS, TASK, ALGO, PRIOR, LOAD, SAMPLER):
    '''Test HPO for in parallel.'''

    # if LOAD is False, create a study from scratch
    if not LOAD:
        # drop the database if exists
        drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
        # create database
        create(munch.Munch({'tag': f'{ALGO}_hpo'}))
        # output_dir
        output_dir = f'./experiments/comparisons/rl/{ALGO}/results'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./experiments/comparisons/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

        fac = ConfigFactory()
        fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
        fac.add_argument("--sampler", type=str, default="TPESampler", help="which sampler to use in HPO.")
        config = fac.merge()
        config.hpo_config.trials = 1
        config.hpo_config.repetitions = 1
        config.sampler = SAMPLER

        hpo(config)
    # if LOAD is True, load a study from a previous HPO study
    else:
        # first, wait a bit untill the HPO study is created
        time.sleep(3)
        # output_dir
        output_dir = f'./experiments/comparisons/rl/{ALGO}/results'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./experiments/comparisons/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                        '--output_dir', output_dir,
                        '--seed', '8',
                        '--use_gpu', 'True'
                        ]

        fac = ConfigFactory()
        fac.add_argument("--load_study", type=bool, default=True, help="whether to load study from a previous HPO.")
        fac.add_argument("--sampler", type=str, default="TPESampler", help="which sampler to use in HPO.")
        config = fac.merge()
        config.hpo_config.trials = 0
        config.sampler = SAMPLER

        hpo(config)

        # delete output_dir
        if os.path.exists(output_dir):
            os.system(f'rm -rf {output_dir}')

        # drop database
        drop(munch.Munch({'tag': f'{ALGO}_hpo'}))

@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
@pytest.mark.parametrize('PRIOR',[''])
def test_hp_perturbation(SYS, TASK, ALGO, PRIOR):
    '''Test Hyperparameter perturbation.
    '''

    # output_dir
    output_dir = f'./experiments/comparisons/rl/{ALGO}/results'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    # drop the database if exists
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        f'./experiments/comparisons/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_hpo_{PRIOR}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '7',
                    '--use_gpu', 'True'
                    ]

    fac = ConfigFactory()
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    config = fac.merge()
    config.hpo_config.hpo = False
    config.hpo_config.perturb_hps = True

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))