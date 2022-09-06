#!/bin/bash

# Model-predictive safety certification of unsafe PPO controller.
python3 ./mpsc_experiment.py --task cartpole --algo ppo --safety_filter linear_mpsc --overrides ./config_overrides/mpsc_config.yaml ./config_overrides/ppo_config.yaml ./config_overrides/cartpole_config.yaml
