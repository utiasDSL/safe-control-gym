#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1
algo="gpmpc_acados"
sys=$2 # cartpole, or quadrotor
task=$3 # stab, or track
sampler=$4 # RandomSampler or TPESampler
FOLDER="./benchmarking_sim/hpo/${algo}"
EXP_NAME="hp_study_normalized_objective" # hp_study_normalized_objective or hp_study_unnormalized_objective
HPO_NAME="normalized_objective" # normalized_objective or unnormalized_objective
OUTPUT_DIR=(${FOLDER}/${EXP_NAME})

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

conda activate safe

# find optimal hyperparameters file
for strat_dir in ${FOLDER}/${HPO_NAME}; do

# Find hyperparameters_ file with the largest return among all seed directories within the run directory
best_hp_file=$(find ${strat_dir}/seed*/ -name "hyperparameters_*.yaml" | awk -F_ '{print $NF,$0}' | sort -n -k1,1 | cut -d' ' -f2- | tail -n 1)
# Find hyperparameters_ file with the smallest return among all seed directories within the run directory
# best_hp_file=$(find ${strat_dir}/seed*/ -name "hyperparameters_*.yaml" | awk -F_ '{print $NF,$0}' | sort -n -k1,1 | cut -d' ' -f2- | head -n 1)
echo "Best hyperparameters file: ${best_hp_file}"

done

# 20 training unseen seeds that are unseen during hpo (hpo only saw seeds in [0, 10000])
seeds=(22403 84244 98825 40417 58454 47838 56715 77833 19880 59009 
       47722 81354 63825 13296 10779 98122 86221 89144 35192 24759)

# Number of concurrent processes to run
parallel_jobs=2
count=0

for seed in "${seeds[@]}"; do

  for hps in "${hp_kind[@]}"; do

    if [ "$hps" == "default" ]; then
        hp_path=''
    elif [ "$hps" == "optimized" ]; then
        hp_path="${best_hp_file}"
    fi

    echo "Training in ${hps} config with seed ${seed}"
    
    # Run the process in the background
    python ./examples/hpo/hpo_experiment.py \
        --algo "${algo}" \
        --task "${sys}" \
        --overrides ./benchmarking_sim/quadrotor/config_overrides/quadrotor_2D_attitude_tracking.yaml \
                    ./benchmarking_sim/quadrotor/config_overrides/gpmpc_acados_quadrotor_2D_attitude_tracking_200.yaml \
        --output_dir "${OUTPUT_DIR}" \
        --opt_hps "${hp_path}" \
        --n_episodes 10 \
        --seed "${seed}" \
        --tag "${hps}" \
        --use_gpu True &

    # Increment count
    count=$((count + 1))

    # Check if we have hit the limit of parallel jobs
    if (( count % parallel_jobs == 0 )); then
        # Wait for all background jobs to finish before continuing
        wait
    fi

  done

done
