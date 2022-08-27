import sys

from competition.getting_started import run as competition_run

def test_competition():
    sys.argv[1:] = ['--overrides', './competition/getting_started.yaml'] 
    competition_run(test=True)
