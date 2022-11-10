#!/bin/bash

# H2 Robust Controller with Gaussian Process Experiment.
python3 ./gp_rc_experiment.py --task quadrotor --algo gp_rc --overrides ./config_gp_rc_quadrotor.yaml --output_dir ./results/gprc_quad --render true --thread 1 --func test