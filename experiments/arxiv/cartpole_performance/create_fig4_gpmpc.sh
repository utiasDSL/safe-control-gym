#!/bin/bash

# _XXX refers to using a prior model with 130, 150, and 300% inertial parameters.
# _150 was used to generate paper results.
# check if folder data exists
if [ -d "./data/cartpole_ctrl_perf/" ]
then
    echo "Directory ./data/ already exists."
    plot_data="./data/cartpole_ctrl_perf/*/trajectory_plot.csv"
else
    # otherwise unzip
    unzip gpmpc_fig_data.zip
    plot_data="./gpmpc_fig4_data.csv"
fi

python3 ./utils/gpmpc_cartpole_control_performance.py --plot_dir ${plot_data}

rm -rf gpmpc_fig6_data.csv
rm -rf gpmpc_fig4_data.csv
