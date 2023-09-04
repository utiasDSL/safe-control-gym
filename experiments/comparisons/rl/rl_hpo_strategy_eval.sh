#!/bin/bash

######## NOTE ########
# 00. Perform HPO (remember to adjust hpo config). 
#     ex: python ./experiments/comparisons/ppo/ppo_experiment.py --func hpo --overrides ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole.yaml ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo.yaml --output_dir ./experiments/comparisons/ppo/hpo --task cartpole --seed 2 --use_gpu True
# 1. Specify the EXP_NAME.
# 2. Specify the CONFIG_PATH. ex: ./experiments/.../hpo/hyperparameters_127.2329.yaml
# 3. Change the num_checkpoints as it will be used for evaluation.
# 4. Change the saving folder.
# 5. Change the --fun in ppo_experiment.py.
#####################

cd ~/safe-control-gym

localOrHost=$1
sampler=$2 # RandomSampler or TPESampler
algo=$3 # ppo, sac, or ddpg
sys=$4 # cartpole, or quadrotor
task=$5 # stab, or track
FOLDER="./experiments/comparisons/rl/${algo}"
EXP_NAME="hpo_strategy_study"
OUTPUT_DIR=(${FOLDER}/${EXP_NAME}_${sampler}_${sys})

# Strategy 1: naive single run
# Strategy 2: naive multiple runs
# Strategy 3: multiple runs w/ CVaR
# Strategy 4: dynamic runs w/ CVaR
# Strategy 5: dynamic runs w/o CVaR
strategies=(1 2 3 4 5)

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

# 50 training unseen seeds that are unseen during hpo (hpo only saw seeds in [0, 10000])
seeds=(22403 84244 98825 40417 58454 47838 56715 77833 19880 59009 
       63064 38354 53430 55046 56811 73802 54158 53335 79389 11887 
       95030 78138 97022 63431 55292 13851 10004 36930 20743 67556 
       89353 93490 92286 94057 32920 34437 29695 80638 84519 12407 
       47722 81354 63825 13296 10779 98122 86221 89144 35192 24759)

# Initialize the strategy_runs array
strategy_runs=()

# Loop through the strategy directories
for strat_dir in ${FOLDER}/hpo/hpo_strategy_study_${sampler}/; do

    # Loop through the run directories
    for run_dir in ${strat_dir}run*/; do
        run_name=$(basename $run_dir)

        # Find the largest hyperparameters_ file among all seed directories within the run directory
        best_hp_file=$(find ${run_dir}seed*/ -name "hyperparameters_*.yaml" | sort -t_ -k2,2n | tail -n 1)

        for seed in "${seeds[@]}"
        do
            echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
            python ./experiments/comparisons/rl/rl_experiment.py \
                    --algo ${algo} \
                    --overrides \
                    ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
                    ./experiments/comparisons/rl/config_overrides/${sys}/${sys}_${task}.yaml \
                    --output_dir $OUTPUT_DIR \
                    --tag ${run_name} \
                    --opt_hps ${best_hp_file} \
                    --task ${sys} --seed $seed --use_gpu True
        done

        print the best hyperparameter file
        echo $best_hp_file
        strategy_runs+=( "$run_name" )
    done
done


for strategy_run in "${strategy_runs[@]}"
do

    algo_seed_dir=(./experiments/comparisons/rl/${algo}/${EXP_NAME}_${sampler}_${sys}/${strategy_run}/seed*)

    for algo_seed_dir in "${algo_seed_dir[@]}"
    do

        echo ${algo_seed_dir}
        python ./experiments/comparisons/rl/eval.py \
                --algo ${algo} \
                --func test_from_checkpoints \
                --restore ${algo_seed_dir} \
                --set_test_seed_as_training_eval \
                --kv_overrides task_config.done_on_out_of_bound=False

    done

done