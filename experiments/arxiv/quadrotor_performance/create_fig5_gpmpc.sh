#!/bin/bash

if [ -d "./data/quad_ctrl_perf/" ]
then
    echo "Directory ./data/ already exists."
    plot_data="./data/quad_ctrl_perf/*/trajectory_plot.csv"
else
    # otherwise unzip
    unzip gpmpc_fig_data.zip
    plot_data="./gpmpc_fig5_data.csv"
fi

python3 ./utils/gpmpc_quadrotor_control_performance.py --plot_dir ${plot_data}

rm -rf gpmpc_fig5_data.csv
rm -rf gpmpc_fig6_data.csv
