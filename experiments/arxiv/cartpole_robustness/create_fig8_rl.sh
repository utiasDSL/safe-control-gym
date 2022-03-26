#!/bin/bash

# check if folder data exists
if [ -d "./data/" ] 
then
    echo "Directory ./data/ already exists." 
else
    # otherwise unzip
    unzip cartpole_robustness_data.zip
fi

# plot robustness w.r.t pole_length 
bash utils/rl_cartpole_robustness.sh - plot_robustness pole_length

# plot robustness w.r.t action white noise
bash utils/rl_cartpole_robustness.sh - plot_robustness act_white