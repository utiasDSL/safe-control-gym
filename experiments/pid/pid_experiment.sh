#!/bin/bash

# PID Experiment.

## Tracking
python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_pid_quadrotor.yaml

## Stabilization
python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_pid_quadrotor_stabilization.yaml