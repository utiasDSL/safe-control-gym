#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1
bo_algo=$2
algo=$3
gamma=$4
n_budget=$5
repeat_eval=$6
seed=$7
n_initial=$8
min_repeat_eval=$9
metric=${10}
num_processes=${11}
run=${12}


FOLDER="/examples/hpo/quadrotor_2D_attitude/${algo}/${run}"


# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'hostx' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

conda activate erahbo

# run bo
BO_FOLDER="../../..${FOLDER}"
python ./risk-averse-hetero-bo/runner_files/erahbo_experiment.py \
        --output_dir ${BO_FOLDER} \
        --BO_algo ${bo_algo} \
        --algo ${algo} \
        --gamma ${gamma} \
        --n_budget ${n_budget} \
        --repeat_eval ${repeat_eval} \
        --seed ${seed} \
        --n_initial ${n_initial} \
        --min_repeat_eval ${min_repeat_eval} \
        --metric ${metric} \
        --max_processes ${num_processes} &

# activate SCG environment
conda deactivate
conda activate safe

# init objective interface
SCG_FOLDER=".${FOLDER}"
python ./examples/hpo/objective_interface.py \
        --output_dir ${SCG_FOLDER} \
        --tag "${bo_algo}" &


