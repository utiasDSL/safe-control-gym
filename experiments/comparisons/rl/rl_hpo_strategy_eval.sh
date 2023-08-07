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

experiment_name=$1
localOrHost=$2
sampler=$3 # RandomSampler or TPESampler
algo=$4 # ppo, sac, or ddpg
task=$5 # cartpole, or quadrotor
FOLDER="./experiments/comparisons/${algo}"
EXP_NAME="hpo_strategy_study"
OUTPUT_DIR=(${FOLDER}/${EXP_NAME}_${sampler}_${task})

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

# optimized hyperparameters from each strategy
opt_hps=(
./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s1/seed12_Jul-30-19-01-14_14ae2aa/hpo/hyperparameters_138.8556.yaml 
./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s2/seed12_Jul-31-02-58-15_14ae2aa/hpo/hyperparameters_133.6558.yaml 
./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s3/seed12_Jul-31-11-38-23_14ae2aa/hpo/hyperparameters_132.5965.yaml 
./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s4/seed12_Jul-31-19-50-06_14ae2aa/hpo/hyperparameters_133.0253.yaml 
./experiments/comparisons/ppo/hpo/hpo_strategy_study_TPESampler/run1_s5/seed8_Aug-02-13-06-09_14ae2aa/hpo/hyperparameters_136.2288.yaml
)

for seed in "${seeds[@]}" 
do

for strategy in "${strategies[@]}"
do

       echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
       python ./experiments/comparisons/rl/rl_experiment.py \
              --algo ${algo} \
              --overrides \
              ./experiments/comparisons/${algo}/config_overrides/${task}/${algo}_${task}_.yaml \
              ./experiments/comparisons/rl/config_overrides/${task}/${task}_stab.yaml \
              --output_dir $OUTPUT_DIR \
              --tag run${experiment_name}_s${strategy} \
              --opt_hps ${opt_hps[${strategy}]} \
              --task ${task} --seed $seed --use_gpu True

done

done

for strategy in "${strategies[@]}"
do

algo_seed_dir=(./experiments/comparisons/${algo}/${EXP_NAME}_${sampler}/run${experiment_name}_s${strategy}/seed*)

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