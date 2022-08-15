import sys
import pytest

from experiments.lqr.lqr_experiment import run


@pytest.mark.parametrize('SYS', ['cartpole', 'quadrotor'])
@pytest.mark.parametrize('TASK',['stabilization', 'tracking'])
@pytest.mark.parametrize('ALGO',['lqr', 'ilqr'])
def test_lqr(SYS, TASK, ALGO):
    sys.argv[1:] = ['--algo', ALGO, '--task', SYS, '--overrides', f'./experiments/lqr/config_overrides/{SYS}/{SYS}_{TASK}.yaml', f'./experiments/lqr/config_overrides/{SYS}/{ALGO}_{SYS}_{TASK}.yaml'] 
    run(gui=False, training=False, n_episodes=None, n_steps=10, save_data=False)
