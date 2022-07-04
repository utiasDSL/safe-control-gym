#!/bin/bash

# Remove previous results.
rm -r -f ./temp-data/
rm -r -f ./data/

# Data Efficiency Plots
python3 ./utils/gpmpc_quadrotor_data_eff.py --algo gp_mpc --task quadrotor --overrides ./config_overrides/gpmpc_quadrotor_data_eff.yaml
# Control Performance plots
python3 ./utils/gpmpc_quadrotor_control_performance.py --algo gp_mpc --task quadrotor --overrides ./config_overrides/gpmpc_quadrotor_control_performance.yaml

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp-data/ ./data/
