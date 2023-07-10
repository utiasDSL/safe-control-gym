import sys, os
import pytest
import munch


from safe_control_gym.utils.configuration import ConfigFactory
from experiments.comparisons.gpmpc.gpmpc_experiment import hpo, train
from safe_control_gym.hyperparameters.database import create, drop


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('TASK',['stab'])
@pytest.mark.parametrize('ALGO',['gp_mpc'])
@pytest.mark.parametrize('PRIOR',['150'])

def test_hpo_gpmpc_cartpole(SYS, TASK, ALGO, PRIOR):

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