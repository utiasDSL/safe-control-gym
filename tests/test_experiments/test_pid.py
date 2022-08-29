import sys

from experiments.pid.pid_experiment import run


def test_pid_stabilization():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_stabilization.yaml']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)

def test_pid_trajectory_tracking():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_tracking.yaml']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)
