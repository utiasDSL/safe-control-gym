#!/bin/bash

if [ -d "./data/quad_impossible_traj_10hz/" ]
then
    echo "Directory ./data/ already exists."
    plot_data="./data/quad_impossible_traj_10hz/*/"
else
    # otherwise unzip
    unzip gpmpc_fig7_data.zip
    plot_data="./fig7_csvs/"
fi
python3 ./utils/gpmpc_quadrotor_impossible_traj.py --plot_dir ${plot_data}

rm -rf fig7_csvs
