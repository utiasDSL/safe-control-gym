import sys

from experiments.pid.pid_experiment import run


def test_2D_pid_stabilization():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_2D/quadrotor_2D_stabilization.yaml']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)

def test_2D_pid_trajectory_tracking():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_2D/quadrotor_2D_tracking.yaml']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)

def test_3D_pid_stabilization():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_3D/quadrotor_3D_stabilization.yaml']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)

def test_3D_pid_trajectory_tracking():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_3D/quadrotor_3D_tracking.yaml']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)

def test_3D_pid_custom_trajectory_tracking():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './experiments/pid/config_overrides/quadrotor_3D/quadrotor_3D_tracking.yaml', '--kv_overrides', 'task_config.task_info.trajectory_type=custom']
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)
