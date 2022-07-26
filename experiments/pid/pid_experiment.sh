#!/bin/bash

# PID Experiment.

## Stabilization
python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_overrides/quadrotor_stabilization.yaml

## Tracking
python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_overrides/quadrotor_tracking.yaml
