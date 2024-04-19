#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1
algo="gp_mpc"
sys=$2 # cartpole, or quadrotor
task=$3 # stab, or track
FOLDER="./examples/hpo/${algo}"
EXP_NAME="hp_study"
OUTPUT_DIR=(${FOLDER}/${EXP_NAME}_${sys})

# Strategy 1: default hps
# Strategy 2: optimized hps
hp_kind=(optimized default)

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

conda activate pr-env

# 20 training unseen seeds that are unseen during hpo (hpo only saw seeds in [0, 10000])
seeds=(22403 84244 98825 40417 58454 47838 56715 77833 19880 59009 
       47722 81354 63825 13296 10779 98122 86221 89144 35192 24759)

for seed in "${seeds[@]}"; do

for hps in "${hp_kind[@]}"; do
    if [ "$hps" == "default" ]; then
        hp_path=''
    elif [ "$hps" == "optimized" ]; then
        hp_path="${FOLDER}/config_overrides/cartpole/optimized_hyperparameters.yaml"
    fi

    echo "Training in ${hps} config"
    python ./examples/hpo/hpo_experiment.py \
        --algo gp_mpc \
        --task "${sys}" \
        --overrides ./examples/hpo/gp_mpc/config_overrides/cartpole/gp_mpc_cartpole_150.yaml \
                    ./examples/hpo/gp_mpc/config_overrides/cartpole/cartpole_stab.yaml \
        --output_dir "${OUTPUT_DIR}" \
        --opt_hps "${hp_path}" \
        --seed "${seed}" \
        --tag "${hps}" \
        --use_gpu True
done

done
