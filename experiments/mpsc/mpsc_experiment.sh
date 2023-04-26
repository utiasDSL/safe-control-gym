#!/bin/bash

SYS='cartpole'
# SYS='quadrotor_2D'

# TASK='stab'
TASK='track'

# ALGO='lqr'
# ALGO='pid'
ALGO='ppo'
# ALGO='sac'

SAFETY_FILTER='linear_mpsc'

MPSC_COST='one_step_cost'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Model-predictive safety certification of an unsafe controller.
python3 ./mpsc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --safety_filter ${SAFETY_FILTER} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SAFETY_FILTER}_${SYS}.yaml \
    --kv_overrides \
        sf_config.cost_function=${MPSC_COST}
