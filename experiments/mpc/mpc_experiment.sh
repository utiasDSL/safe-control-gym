#!/bin/bash

# MPC Experiment.

ENV="cartpole"
# ENV="quadrotor"

TASK="stabilization"
# TASK="tracking"

# ALGO="linear_mpc"
# ALGO="tube_mpc"
ALGO="lqr"

python3 ./mpc_experiment.py --task ${ENV} --algo ${ALGO} --overrides ./config_overrides/${TASK}/${ENV}.yaml ./config_overrides/${TASK}/${ALGO}_${ENV}.yaml
