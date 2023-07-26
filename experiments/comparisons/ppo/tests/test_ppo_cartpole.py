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
                    '--tag', 's1',
                    '--opt_hps',
                    './experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s1/seed8_Jul-19-16-29-41_b40566c/hpo/hyperparameters_139.8787.yaml',
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