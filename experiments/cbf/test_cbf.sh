#!/bin/bash

TASK="cartpole"

ALGO="ppo"
# ALGO="sac"

# SAFETY_FILTER="cbf"
SAFETY_FILTER="cbf_nn"

# Model-predictive safety certification of an unsafe controller.
python3 ./cbf_experiment.py --task ${TASK} --algo ${ALGO} --safety_filter ${SAFETY_FILTER} --overrides ./config_overrides/${TASK}_config.yaml ./config_overrides/${ALGO}_config.yaml ./config_overrides/cbf_config.yaml
