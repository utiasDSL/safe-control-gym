#!/bin/bash

# LQR Controller on Cartpole experiment.
python3 ./lqr_experiment.py --task quadrotor --algo lqr --overrides ./config_lqr_quadrotor.yaml --output_dir ./results/lqr_quad --render true --thread 1 --func test