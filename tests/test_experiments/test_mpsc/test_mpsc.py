import sys
import pytest

from experiments.mpsc.mpsc_single_experiment import run


@pytest.mark.parametrize("SYS,TASK,ALGO", [
    ('cartpole', 'stab', 'lqr'), 
    ('cartpole', 'stab', 'ppo'), 
    ('cartpole', 'stab', 'sac'), 
    ('cartpole', 'track', 'lqr'), 
    ('quadrotor', 'stab', 'pid'), 
    ('quadrotor', 'track', 'pid')
])
def test_mpsc(SYS, TASK, ALGO):
    sys.argv[1:] = [
        '--task', SYS, 
        '--algo', ALGO, 
        '--safety_filter', 'mpsc', 
        '--overrides', 
            f'./experiments/mpsc/config_overrides/{SYS}/{SYS}_{TASK}.yaml', 
            f'./experiments/mpsc/config_overrides/{SYS}/{ALGO}_{SYS}.yaml', 
            f'./experiments/mpsc/config_overrides/{SYS}/mpsc_{SYS}.yaml'
        ] 
    run(plot=False, max_steps=10, curr_path='./experiments/mpsc')