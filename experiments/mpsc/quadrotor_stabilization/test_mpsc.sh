#!/bin/bash

TASK="quadrotor"

ALGO="pid"
# ALGO="ppo"
# ALGO="sac"

SAFETY_FILTER="mpsc"

# Model-predictive safety certification of an unsafe controller.
python3 ./${SAFETY_FILTER}_${TASK}_experiment.py --task ${TASK} --algo ${ALGO} --safety_filter ${SAFETY_FILTER}_sf --overrides ./config_overrides/${TASK}_config.yaml ./config_overrides/${ALGO}_config.yaml ./config_overrides/${SAFETY_FILTER}_config.yaml
