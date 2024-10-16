#!/bin/bash

# MPC and Linear MPC Experiment.

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_3D'

#TASK='stabilization'
TASK='tracking'

# ALGO='mpc'
# ALGO='mpc_acados'
ALGO='linear_mpc'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./mpc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}.yaml
