#!/bin/bash

# LQR Experiment.

# SYS='cartpole'
SYS='quadrotor_2D'
# SYS='quadrotor_3D'

# TASK='stabilization'
TASK='tracking'

# ALGO='lqr'
ALGO='ilqr'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./lqr_experiment.py --task ${SYS_NAME} --algo ${ALGO} --overrides ./config_overrides/${SYS}/${SYS}_${TASK}.yaml ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}.yaml
