#!/bin/bash

# Model-predictive safety certification of unsafe PPO controller.
python3 ./mpsc_experiment.py --task cartpole --algo mpsc --overrides ./config_overrides/config_mpsc_cartpole.yaml
