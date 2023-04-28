import sys
import pytest

from examples.rl.rl_experiment import run


@pytest.mark.parametrize('SYS', ['cartpole', 'quadrotor_2D', 'quadrotor_3D'])
@pytest.mark.parametrize('TASK',['stab', 'track'])
@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
def test_rl(SYS, TASK, ALGO):
    SYS_NAME = 'quadrotor' if 'quadrotor' in SYS else SYS
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS_NAME,
                    '--overrides',
                        f'./examples/rl/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                        f'./examples/rl/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
                    '--kv_overrides',
                        'algo_config.training=False',
                    ]
    run(gui=False, n_episodes=None, n_steps=10, curr_path='./examples/rl')
