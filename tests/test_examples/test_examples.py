import sys

from examples.tracking import run as tracking_run
from examples.verbose_api import run as verbose_run

def test_tracking():
    sys.argv[1:] = ['--overrides', './examples/tracking.yaml'] 
    tracking_run(gui=False, max_steps=10)

def test_verbose_api():
    sys.argv[1:] = ['--overrides', './examples/verbose_api.yaml']
    verbose_run()