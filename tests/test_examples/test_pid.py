import sys
import pytest

from examples.pid.pid_experiment import run


@pytest.mark.parametrize('SYS', ['quadrotor_2D', 'quadrotor_3D'])
@pytest.mark.parametrize('TASK',['stabilization', 'tracking'])
def test_pid(SYS, TASK):
    sys.argv[1:] = ['--algo', 'pid',
                    '--task', 'quadrotor',
                    '--overrides',
                        f'./examples/pid/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
                    ]
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)


def test_3D_pid_custom_trajectory_tracking():
    sys.argv[1:] = ['--algo', 'pid',
                    '--task', 'quadrotor',
                    '--overrides',
                        './examples/pid/config_overrides/quadrotor_3D/quadrotor_3D_tracking.yaml',
                    '--kv_overrides',
                        'task_config.task_info.trajectory_type=custom'
                    ]
    run(gui=False, n_episodes=None, n_steps=10, save_data=False)
