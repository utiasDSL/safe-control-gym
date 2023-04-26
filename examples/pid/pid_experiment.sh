#!/bin/bash

# PID Experiment.

# SYS='quadrotor_2D'
SYS='quadrotor_3D'

# TASK='stabilization'
TASK='tracking'

# TRAJ_TYPE='figure8'
# TRAJ_TYPE='circle'
# TRAJ_TYPE='square'
TRAJ_TYPE='custom'

python3 ./pid_experiment.py \
    --task quadrotor \
    --algo pid \
    --overrides \
        ./config_overrides/pid.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
    --kv_overrides \
        task_config.task_info.trajectory_type=${TRAJ_TYPE}
