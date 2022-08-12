import sys

from examples.tracking import run as tracking_run
from examples.quad_3D import run as quad_3D_run
from examples.verbose_api import run as verbose_run

def test_tracking():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './examples/tracking.yaml'] 
    tracking_run(gui=False, n_episodes=None, n_steps=10)

def test_quad_3D_custom():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './examples/quad_3D.yaml'] 
    quad_3D_run(gui=False, n_episodes=None, n_steps=10, custom_trajectory=True)

def test_quad_3D_figure8():
    sys.argv[1:] = ['--task', 'quadrotor', '--algo', 'pid', '--overrides', './examples/quad_3D.yaml'] 
    quad_3D_run(gui=False, n_episodes=None, n_steps=10, custom_trajectory=False)

def test_verbose_api_cartpole():
    sys.argv[1:] = ['--task', 'cartpole', '--overrides', './examples/verbose_api.yaml']
    verbose_run()

def test_verbose_api_quadrotor():
    sys.argv[1:] = ['--task', 'quadrotor', '--overrides', './examples/verbose_api.yaml']
    verbose_run()
