#!/bin/bash


# check if folder data exists
if [ -d "./data/cartpole_white_noise_robust/" ]
then

    echo "Directory ./data/ already exists."
    plot_data_white_noise="./data/cartpole_white_noise_robust/*/rmse_robust_plot.csv"
else
    # otherwise unzip
    unzip gpmpc_fig8_data.zip
    plot_data_white_noise="./gpmpc_fig8_input_noise_data.csv"
fi

python3 ./utils/gpmpc_cartpole_input_white_noise_robustness.py --plot_dir ${plot_data_white_noise}

rm -rf gpmpc_fig8_input_noise_data.csv
rm -rf gpmpc_fig8_length_data.csv

if [ -d "./data/cartpole_pole_length_robust" ]
then

    echo "Directory ./data/ already exists."
    plot_data_length="./data/cartpole_pole_length_robust/*/rmse_robust_plot.csv"
else
    # otherwise unzip
    unzip gpmpc_fig8_data.zip
    plot_data_length="./gpmpc_fig8_length_data.csv"
fi

python3 ./utils/gpmpc_cartpole_pole_length_robustness.py --plot_dir ${plot_data_length}

rm -rf gpmpc_fig8_input_noise_data.csv
rm -rf gpmpc_fig8_length_data.csv
