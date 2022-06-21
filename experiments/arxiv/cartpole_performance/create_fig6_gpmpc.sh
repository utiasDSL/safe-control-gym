#!/bin/bash

# _XXX refers to using a prior model with 130, 150, and 300% inertial parameters.
# _150 was used to generate paper results.
# check if folder data exists
if [ -d "./data/cartpole_data_eff/" ]
then
    echo "Directory ./data/ already exists."
    plot_data="./data/cartpole_data_eff/*/figs/avg_rmse_cost_learning_curve.csv"
else
    # otherwise unzip
    unzip gpmpc_fig_data.zip
    plot_data="./gpmpc_fig6_data.csv"
fi

python3 ./utils/gpmpc_cartpole_data_eff.py --plot_dir ${plot_data}

# Cleanup csv data
rm -r -f gpmpc_fig4_data.csv
rm -r -f gpmpc_fig6_data.csv
