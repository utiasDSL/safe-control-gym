#!/bin/bash

# PID Experiment.

## Tracking
python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_overrides/quadrotor_stabilization.yaml

## Stabilization
python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_overrides/quadrotor_tracking.yaml