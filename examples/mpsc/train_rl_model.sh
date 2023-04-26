#!/bin/bash

SYS='cartpole'
# SYS='quadrotor'

TASK='stab'
# TASK='track'

ALGO='ppo'
# ALGO='sac'

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/

# Train the unsafe controller/agent.
python3 ../execute_rl_controller.py --algo ${ALGO} --task ${SYS} --overrides ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml ./config_overrides/${SYS}/${SYS}_${TASK}.yaml --output_dir ./ \
                    --tag unsafe_rl_temp_data/ --seed 2

# Move the newly trained unsafe model.
mv ./unsafe_rl_temp_data/seed2_*/model_latest.pt ./models/${ALGO}_model_${SYS}_${TASK}.pt

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/
