import sys

from experiments.pid.pid_experiment import run

def test_pid_trajectory_tracking():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './experiments/pid/config_pid_quadrotor.yaml'] 
    run(gui=False, max_steps=10)

def test_pid_stabilization():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './experiments/pid/config_pid_quadrotor_stabilization.yaml'] 
    run(gui=False, max_steps=10)