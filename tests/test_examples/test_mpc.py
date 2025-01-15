import sys

import pytest

from examples.mpc.mpc_experiment import run


@pytest.mark.parametrize('SYS',  ['cartpole', 'quadrotor_2D', 'quadrotor_3D'])
@pytest.mark.parametrize('TASK', ['stabilization', 'tracking'])
@pytest.mark.parametrize('ALGO', ['mpc', 'linear_mpc'])
def test_lqr(SYS, TASK, ALGO):
    SYS_NAME = 'quadrotor' if 'quadrotor' in SYS else SYS
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS_NAME,
                    '--overrides',
                        f'./examples/mpc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./examples/mpc/config_overrides/{SYS}/{ALGO}_{SYS}_{TASK}.yaml',
                    '--kv_overrides',
                        'algo_config.max_iterations=2'
                    ]
    run(gui=False, plot=False, n_episodes=None, n_steps=10, save_data=False)
