#!/bin/bash

# Removed the temporary data used to train the new unsafe PPO model.
rm -r -f ./unsafe_ppo_temp_data/

# Train the unsafe PPO controller/agent.
python3 ../../main.py --algo ppo --task cartpole --overrides ./config_overrides/unsafe_ppo_config.yaml --output_dir ./ \
                    --tag unsafe_ppo_temp_data/ --kv_overrides task_config.ctrl_freq=10 task_config.pyb_freq=1000 --seed 2

# Backup the unsafe PPO model.
cp ./unsafe_ppo_model/unsafe_ppo_model_30000.pt ./unsafe_ppo_model/bak_unsafe_ppo_model_30000.pt 

# Move the newly trained unsafe PPO model.
mv ./unsafe_ppo_temp_data/seed2_*/model_latest.pt ./unsafe_ppo_model/unsafe_ppo_model_30000.pt 

# Removed the temporary data used to train the new unsafe PPO model.
rm -r -f ./unsafe_ppo_temp_data/
