import sys
import pytest

from experiments.mpsc.mpsc_single_experiment import run


@pytest.mark.parametrize("SYS,TASK,ALGO,SAFETY_FILTER", [
    ('cartpole', 'stab', 'lqr', 'mpsc'), 
    ('cartpole', 'stab', 'ppo', 'mpsc'), 
    ('cartpole', 'stab', 'sac', 'mpsc'), 
    ('cartpole', 'track', 'lqr', 'mpsc'), 
    ('quadrotor', 'stab', 'pid', 'mpsc'), 
    ('quadrotor', 'track', 'pid', 'mpsc'),
])
def test_mpsc(SYS, TASK, ALGO, SAFETY_FILTER):
    sys.argv[1:] = [
        '--task', SYS, 
        '--algo', ALGO, 
        '--safety_filter', SAFETY_FILTER, 
        '--overrides', 
            f'./experiments/mpsc/config_overrides/{SYS}/{SYS}_{TASK}.yaml', 
            f'./experiments/mpsc/config_overrides/{SYS}/{ALGO}_{SYS}.yaml', 
            f'./experiments/mpsc/config_overrides/{SYS}/mpsc_{SYS}.yaml'
        ] 
    run(plot=False, max_steps=10, curr_path='./experiments/mpsc')