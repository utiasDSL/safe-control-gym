#!/bin/bash

# LQR Controller on Cartpole experiment.
python3 ./ilqr_experiment.py --task quadrotor --algo ilqr --overrides ./config_ilqr_quadrotor.yaml --output_dir ./results/ilqr_quad --render true --thread 1 --func test