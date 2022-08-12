#!/bin/bash

# LQR Experiment.

# SYS='cartpole'
SYS='quadrotor'

# TASK='stabilization'
TASK='tracking'

# ALGO='lqr'
ALGO='ilqr'

python3 ./lqr_experiment.py --task ${SYS} --algo ${ALGO} --overrides ./config_overrides/${SYS}_${TASK}.yaml ./config_overrides/${ALGO}_config.yaml
