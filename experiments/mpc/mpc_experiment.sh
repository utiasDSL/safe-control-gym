#!/bin/bash

# MPC Experiment.

# ENV="cartpole"
ENV="quadrotor"

TASK="tracking"
# TASK="tracking"

ALGO="linear_mpc"
# ALGO="tube_mpc"

python3 ./mpc_experiment.py --task ${ENV} --algo ${ALGO} --overrides ./config_overrides/${TASK}/${ENV}_config.yaml ./config_overrides/${TASK}/${ALGO}_config.yaml
