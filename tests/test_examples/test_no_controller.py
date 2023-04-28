import sys

from examples.no_controller.verbose_api import run as verbose_run


def test_verbose_api_cartpole():
    sys.argv[1:] = ['--task', 'cartpole', '--overrides', './examples/no_controller/verbose_api.yaml']
    verbose_run()


def test_verbose_api_quadrotor():
    sys.argv[1:] = ['--task', 'quadrotor', '--overrides', './examples/no_controller/verbose_api.yaml']
    verbose_run()
