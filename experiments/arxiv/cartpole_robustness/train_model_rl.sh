#!/bin/bash

# Remove previous results.
rm -r -f ./temp_data/
rm -r -f ./data/

# train models 
bash utils/rl_cartpole_robustness.sh ppo train
bash utils/rl_cartpole_robustness.sh ppo train_dr
bash utils/rl_cartpole_robustness.sh rarl train_rarl
bash utils/rl_cartpole_robustness.sh rap train_rap

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp_data/ ./data/
