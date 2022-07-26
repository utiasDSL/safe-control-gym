import sys

from experiments.pid.pid_experiment import run


def test_pid_stabilization():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './experiments/pid/config_overrides/quadrotor_stabilization.yaml'] 
    run(gui=False, max_steps=10)

def test_pid_trajectory_tracking():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './experiments/pid/config_overrides/quadrotor_tracking.yaml'] 
    run(gui=False, max_steps=10)
