import sys

from tests.scripts.tracking import run as tracking_run
from tests.scripts.verbose_api import run as verbose_run

def test_tracking():
    sys.argv[1:] = ['--overrides', './tests/scripts/tracking.yaml'] 
    tracking_run(gui=False, max_steps=10)

def test_verbose_api_quadrotor():
    sys.argv[1:] = ['--overrides', './tests/scripts/verbose_api.yaml', '--task', 'quadrotor']
    verbose_run()
