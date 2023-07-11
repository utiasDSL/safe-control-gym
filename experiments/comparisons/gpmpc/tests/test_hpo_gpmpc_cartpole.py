import sys, os
import pytest
import munch
import time


from safe_control_gym.utils.configuration import ConfigFactory
from experiments.comparisons.gpmpc.gpmpc_experiment import hpo, train
from safe_control_gym.hyperparameters.database import create, drop


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['gp_mpc'])
@pytest.mark.parametrize('PRIOR',['150'])
def test_hpo_gpmpc_cartpole(SYS, TASK, ALGO, PRIOR):
    '''Test HPO for gp_mpc on cartpole stab task for one single trial
        (create a study from scratch)
    '''

    # drop the database if exists
    drop(munch.Munch({'tag': 'gp_mpc_hpo'}))
    # create database
    create(munch.Munch({'tag': 'gp_mpc_hpo'}))
    # output_dir
    output_dir = './experiments/comparisons/gpmpc/results'
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                        f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                    '--output_dir', output_dir,
                    ]

    fac = ConfigFactory()
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    config = fac.merge()
    config.algo_config.num_test_episodes_per_epoch = 1
    config.algo_config.num_train_episodes_per_epoch = 1
    config.hpo_config.trials = 1

    hpo(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': 'gp_mpc_hpo'}))

@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['gp_mpc'])
@pytest.mark.parametrize('PRIOR',['150'])
@pytest.mark.parametrize('LOAD', [False, True])
def test_hpo_gpmpc_cartpole_parallelism(SYS, TASK, ALGO, PRIOR, LOAD):
    '''Test HPO for gp_mpc on cartpole stab task in parallel.'''

    # if LOAD is False, create a study from scratch
    if not LOAD:
        # drop the database if exists
        drop(munch.Munch({'tag': 'gp_mpc_hpo'}))
        # create database
        create(munch.Munch({'tag': 'gp_mpc_hpo'}))
        # output_dir
        output_dir = './experiments/comparisons/gpmpc/results'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                        '--output_dir', output_dir,
                        '--seed', '7',
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
        output_dir = './experiments/comparisons/gpmpc/results'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS,
                        '--overrides',
                            f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                            f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                            f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_hpo.yaml',
                        '--output_dir', output_dir,
                        '--seed', '8',
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
        drop(munch.Munch({'tag': 'gp_mpc_hpo'}))


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['gp_mpc'])
@pytest.mark.parametrize('PRIOR',['150'])
def test_train_gpmpc_cartpole(SYS, TASK, ALGO, PRIOR):
    '''Test training for gp_mpc on cartpole stab task given a set of hyperparameters.
    '''

    # drop the database if exists
    drop(munch.Munch({'tag': 'gp_mpc_hpo'}))
    # create database
    create(munch.Munch({'tag': 'gp_mpc_hpo'}))
    # output_dir
    output_dir = './experiments/comparisons/gpmpc/results'
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--overrides',
                        f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./experiments/comparisons/gpmpc/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                    '--output_dir', output_dir,
                    '--seed', '10810330'
                    ]
    
    fac = ConfigFactory()
    fac.add_argument("--opt_hps", type=str, default="", help="yaml file as a result of HPO.")
    config = fac.merge()

    # define hps
    hps = {'horizon': 35, 
           'kernel': 'RBF', 
           'n_ind_points': 40, 'num_epochs': 5, 
           'num_samples': 75, 
           'optimization_iterations': [2800, 2800, 2800, 2800], 
           'learning_rate': [0.023172075157730145, 0.023172075157730145, 0.023172075157730145, 0.023172075157730145]}

    for hp in hps:
        if isinstance(config.algo_config[hp], list) and not isinstance(hps[hp], list):
            config.algo_config[hp] = [hps[hp]] * len(config.algo_config[hp])
        else:
            config.algo_config[hp] = hps[hp]

    train(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    # drop database
    drop(munch.Munch({'tag': 'gp_mpc_hpo'}))