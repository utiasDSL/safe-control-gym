#!/bin/bash

TASK="cartpole"

ALGO="ppo"
# ALGO="sac"

SAFETY_FILTER="mpsc"

TEST_TYPE=""
# TEST_TYPE="stat_"

# Model-predictive safety certification of an unsafe controller.
python3 ./${SAFETY_FILTER}_${TASK}_${TEST_TYPE}experiment.py --task ${TASK} --algo ${ALGO} --safety_filter ${SAFETY_FILTER}_sf --overrides ./config_overrides/${TASK}_config.yaml ./config_overrides/${ALGO}_config.yaml ./config_overrides/${SAFETY_FILTER}_config.yaml
