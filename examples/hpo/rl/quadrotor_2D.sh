#!/bin/bash

cd ~/safe-control-gym
source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
conda activate pr-env

python ./examples/hpo/hpo_experiment.py \
--algo ppo \
--task quadrotor \
--overrides ./examples/hpo/rl/config_overrides/quadrotor_2D_attitude/quadrotor_2D_attitude_track.yaml \
            ./examples/hpo/rl/ppo/config_overrides/quadrotor_2D_attitude/ppo_quadrotor_2D_attitude_.yaml \
--output_dir ./examples/hpo/results --n_episodes 10 --tag 2D_attitude --opt_hps '' --seed 6 --use_gpu True

python ./examples/hpo/hpo_experiment.py \
--algo ppo \
--task quadrotor \
--overrides ./examples/hpo/rl/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml \
            ./examples/hpo/rl/ppo/config_overrides/quadrotor_2D/ppo_quadrotor_2D_.yaml \
--output_dir ./examples/hpo/results --n_episodes 10 --tag 2D --opt_hps '' --seed 6 --use_gpu True

python ./examples/hpo/hpo_experiment.py \
--algo sac \
--task quadrotor \
--overrides ./examples/hpo/rl/config_overrides/quadrotor_2D_attitude/quadrotor_2D_attitude_track.yaml \
            ./examples/hpo/rl/sac/config_overrides/quadrotor_2D_attitude/sac_quadrotor_2D_attitude_.yaml \
--output_dir ./examples/hpo/results --n_episodes 10 --tag 2D_attitude --opt_hps '' --seed 6 --use_gpu True

python ./examples/hpo/hpo_experiment.py \
--algo sac \
--task quadrotor \
--overrides ./examples/hpo/rl/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml \
            ./examples/hpo/rl/sac/config_overrides/quadrotor_2D/sac_quadrotor_2D_.yaml \
--output_dir ./examples/hpo/results --n_episodes 10 --tag 2D --opt_hps '' --seed 6 --use_gpu True