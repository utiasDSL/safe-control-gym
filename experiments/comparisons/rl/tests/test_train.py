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
def test_train(SYS, TASK, ALGO, PRIOR):
    '''Test training given a set of hyperparameters.
    '''
    
    # output_dir
    output_dir = f'./experiments/comparisons/rl/{ALGO}/results'
    # delete output_dir if exists
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
                    '--output_dir', output_dir,
                    '--tag', 's1',
                    '--opt_hps',
                    './experiments/comparisons/rl/ppo/hpo/hpo_strategy_study_TPESampler/run1_s1/seed8_Jul-19-16-29-41_b40566c/hpo/hyperparameters_139.8787.yaml',
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
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))