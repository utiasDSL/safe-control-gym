#!/bin/bash

# Remove previous results.
rm -r -f ./temp-data/
rm -r -f ./data/

# _XXX refers to using a prior model with 130, 150, and 300% inertial parameters.
# _150 was used to generate paper results.

#python3 ./utils/gpmpc_cartpole_input_white_noise_robustness.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_input_white_noise_robustness_130.yaml
python3 ./utils/gpmpc_cartpole_input_white_noise_robustness.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_input_white_noise_robustness_150.yaml
#python3 ./utils/gpmpc_cartpole_input_white_noise_robustness.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_input_white_noise_robustness_300.yaml
#python3 ./utils/gpmpc_cartpole_pole_length_robustness.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_pole_length_robustness_130.yaml
python3 ./utils/gpmpc_cartpole_pole_length_robustness.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_pole_length_robustness_150.yaml
#python3 ./utils/gpmpc_cartpole_pole_length_robustness.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_pole_length_robustness_300.yaml

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp-data/ ./data/
