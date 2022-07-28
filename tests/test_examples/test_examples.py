import sys

from examples.tracking import run as tracking_run
from examples.verbose_api import run as verbose_run

def test_tracking():
    sys.argv[1:] = ['--algo', 'pid', '--task', 'quadrotor', '--overrides', './examples/tracking.yaml'] 
    tracking_run(gui=False, n_episodes=None, n_steps=10)

def test_verbose_api_cartpole():
    sys.argv[1:] = ['--task', 'cartpole', '--overrides', './examples/verbose_api.yaml']
    verbose_run()

def test_verbose_api_quadrotor():
    sys.argv[1:] = ['--task', 'quadrotor', '--overrides', './examples/verbose_api.yaml']
    verbose_run()
