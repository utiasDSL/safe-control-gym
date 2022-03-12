#!/bin/bash

# check if folder data exists
if [ -d "./data/" ] 
then
    echo "Directory ./data/ already exists." 
else
    # otherwise unzip
    unzip cartpole_robustness_data.zip
fi

# evaluate on different pole lengths 
bash utils/rl_cartpole_robustness.sh ppo test_params "seed*"
bash utils/rl_cartpole_robustness.sh ppo_dr_pole_length test_params "*/seed*"
bash utils/rl_cartpole_robustness.sh rarl test_params "scale*/seed*"
bash utils/rl_cartpole_robustness.sh rap test_params "scale*/seed*"

# evaluate on different scales of action white noise
bash utils/rl_cartpole_robustness.sh ppo test_disturbances "seed*"
bash utils/rl_cartpole_robustness.sh ppo_dr_pole_length test_disturbances "*/seed*"
bash utils/rl_cartpole_robustness.sh rarl test_disturbances "scale*/seed*"
bash utils/rl_cartpole_robustness.sh rap test_disturbances "scale*/seed*"
