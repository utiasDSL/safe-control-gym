#!/bin/bash

if [ -d "./data/quad_data_eff/" ]
then
    echo "Directory ./data/ already exists."
    plot_data="./data/quad_data_eff/*/figs/rmse_xz_error_learning_curve.csv"
else
    # otherwise unzip
    unzip gpmpc_fig_data.zip
    plot_data="./gpmpc_fig6_data.csv"
fi

python3 ./utils/gpmpc_quadrotor_data_eff.py --plot_dir ${plot_data}

rm -rf gpmpc_fig5_data.csv
rm -rf gpmpc_fig6_data.csv
