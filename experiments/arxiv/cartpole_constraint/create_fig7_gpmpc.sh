#!/bin/bash
# _XXX refers to using a prior model with 130, 150, and 300% inertial parameters.
# _150 was used to generate paper results.
# check if folder data exists
if [ -d "./data/cartpole_constraint_plot/" ]
then
    echo "Directory ./data/ already exists."
    plot_data="./data/cartpole_constraint_plot/*/figs/number_viol.csv"
else
    # otherwise unzip
    unzip gpmpc_fig_data.zip
    plot_data="./gpmpc_number_viols.csv"
fi

python3 ./utils/gpmpc_cartpole_constraint.py --plot_dir ${plot_data}

rm -rf gpmpc_number_viols.csv
