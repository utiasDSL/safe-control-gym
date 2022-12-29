#!/bin/bash

# Remove previous results.
rm -r -f ./temp-data/
rm -r -f ./data/

python3 ./utils/DELENDA_epoch_experiment_example.py --algo gp_mpc --task quadrotor --overrides ./config_overrides/gpmpc_impossible_traj.yaml

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp-data/ ./data/
