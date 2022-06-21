#!/bin/bash

# SYS="cartpole"
SYS="quadrotor"

TASK="stab"
# TASK="track"

# ALGO="lqr"
ALGO="pid"
# ALGO="ppo"
# ALGO="sac"

SAFETY_FILTER="mpsc"

EXPERIMENT="single"
# EXPERIMENT="stat"

# Model-predictive safety certification of an unsafe controller.
python3 ./mpsc_${EXPERIMENT}_experiment.py --task ${SYS} --algo ${ALGO} --safety_filter ${SAFETY_FILTER} --overrides ./config_overrides/${SYS}/${SYS}_${TASK}.yaml ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml ./config_overrides/${SYS}/mpsc_${SYS}.yaml
