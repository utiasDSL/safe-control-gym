#!/bin/bash

# LQR Experiment.

#SYS='cartpole'
#SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
#SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

#ALGO='lqr'
ALGO='ilqr'
#ALGO='ilqr_c'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./lqr_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}.yaml
