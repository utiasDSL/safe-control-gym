#!/bin/bash

# MPC Experiment.

ENV="cartpole"

# TASK="stabilization"
TASK="tracking"

# ALGO="linear_mpc"
# ALGO="tube_mpc"
ALGO="lqr"

python3 ./mpc_experiment.py --task ${ENV} --algo ${ALGO} --overrides ./config_overrides/${ENV}_${TASK}.yaml ./config_overrides/${ALGO}_${ENV}.yaml
