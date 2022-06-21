#!/bin/bash

# Remove previous results.
rm -r -f ./temp-data/
rm -r -f ./data/

# _XXX refers to using a prior model with 130, 150, and 300% inertial parameters.
# _150 was used to generate paper results.

# Creates the cartpole_data_eff for fig 6.
#python3 ./utils/gpmpc_cartpole_data_eff.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_data_eff_300.yaml
python3 ./utils/gpmpc_cartpole_data_eff.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_data_eff_150.yaml
#python3 ./utils/gpmpc_cartpole_data_eff.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_data_eff_130.yaml

# Creates the cartpole_ctrl_performance directory for fig 4.
#python3 ./utils/gpmpc_cartpole_control_performance.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_control_performance_130.yaml
python3 ./utils/gpmpc_cartpole_control_performance.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_control_performance_150.yaml
#python3 ./utils/gpmpc_cartpole_control_performance.py --algo gp_mpc --task cartpole --overrides ./config_overrides/gpmpc_cartpole_control_performance_300.yaml

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp-data/ ./data/
