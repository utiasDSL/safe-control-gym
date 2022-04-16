#!/bin/bash

# MPC Experiment.

## Tracking
python3 ./mpc_experiment.py --task quadrotor --algo linear_mpc --overrides ./mpc_quad.yaml

python3 ./mpc_experiment.py --task quadrotor --algo linear_mpc --overrides ./mpc_stab.yaml

python3 ./mpc_experiment.py --task quadrotor --algo tube_mpc --overrides ./tube_mpc_stab.yaml