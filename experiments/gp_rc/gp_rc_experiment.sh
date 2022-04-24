#!/bin/bash

# H2 Robust Controller with Gaussian Process Experiment.
python3 ./gp_rc_experiment.py --task cartpole --algo gp_rc --overrides ./config_gp_rc_cartpole.yaml