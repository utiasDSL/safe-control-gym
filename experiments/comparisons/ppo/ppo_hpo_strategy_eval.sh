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

FOLDER="./experiments/comparisons/ppo"
EXP_NAME="hpo_strategy_study"
experiment_name=$1
localOrHost=$2
sampler=$3 # RandomSampler or TPESampler
OUTPUT_DIR=(${FOLDER}/${EXP_NAME}_${sampler})

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host2' ]; then
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

######## Eval Strategy 1: naive single run ########
for seed in "${seeds[@]}" 
do

       echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
       python ./experiments/comparisons/ppo/ppo_experiment.py \
              --overrides \
              ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
              ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
              --output_dir $OUTPUT_DIR \
              --tag run${experiment_name}_s1 \
              --opt_hps ./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s1/seed12_Jul-30-19-01-14_14ae2aa/hpo/hyperparameters_138.8556.yaml \
              --task cartpole --seed $seed --use_gpu True

done

algo_seed_dir=(./experiments/comparisons/ppo/${EXP_NAME}_${sampler}/run${experiment_name}_s1/seed*)

for algo_seed_dir in "${algo_seed_dir[@]}"
do

       echo ${algo_seed_dir}
       python ./experiments/comparisons/ppo/eval.py \
              --func test_from_checkpoints \
              --restore ${algo_seed_dir} \
              --set_test_seed_as_training_eval \
              --kv_overrides task_config.done_on_out_of_bound=False

done

######## Eval Strategy 2: naive multiple runs ########
for seed in "${seeds[@]}" 
do

       echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
       python ./experiments/comparisons/ppo/ppo_experiment.py \
              --overrides \
              ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
              ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
              --output_dir $OUTPUT_DIR \
              --tag run${experiment_name}_s2 \
              --opt_hps ./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s2/seed12_Jul-31-02-58-15_14ae2aa/hpo/hyperparameters_133.6558.yaml \
              --task cartpole --seed $seed --use_gpu True

done

algo_seed_dir=(./experiments/comparisons/ppo/${EXP_NAME}_${sampler}/run${experiment_name}_s2/seed*)

for algo_seed_dir in "${algo_seed_dir[@]}"
do

       echo ${algo_seed_dir}
       python ./experiments/comparisons/ppo/eval.py \
              --func test_from_checkpoints \
              --restore ${algo_seed_dir} \
              --set_test_seed_as_training_eval \
              --kv_overrides task_config.done_on_out_of_bound=False

done

######## Eval Strategy 3: multiple runs w/ CVaR ########
for seed in "${seeds[@]}" 
do

       echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
       python ./experiments/comparisons/ppo/ppo_experiment.py \
              --overrides \
              ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
              ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
              --output_dir $OUTPUT_DIR \
              --tag run${experiment_name}_s3 \
              --opt_hps ./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s3/seed12_Jul-31-11-38-23_14ae2aa/hpo/hyperparameters_132.5965.yaml \
              --task cartpole --seed $seed --use_gpu True

done

algo_seed_dir=(./experiments/comparisons/ppo/${EXP_NAME}_${sampler}/run${experiment_name}_s3/seed*)

for algo_seed_dir in "${algo_seed_dir[@]}"
do

       echo ${algo_seed_dir}
       python ./experiments/comparisons/ppo/eval.py \
              --func test_from_checkpoints \
              --restore ${algo_seed_dir} \
              --set_test_seed_as_training_eval \
              --kv_overrides task_config.done_on_out_of_bound=False

done

######## Eval Strategy 4: dynamic runs w/ CVaR ########
for seed in "${seeds[@]}" 
do

       echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
       python ./experiments/comparisons/ppo/ppo_experiment.py \
              --overrides \
              ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
              ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
              --output_dir $OUTPUT_DIR \
              --tag run${experiment_name}_s4 \
              --opt_hps ./experiments/comparisons/ppo/hpo/hpo_strategy_study_RandomSampler/run3_s4/seed12_Jul-31-19-50-06_14ae2aa/hpo/hyperparameters_133.0253.yaml \
              --task cartpole --seed $seed --use_gpu True

done

algo_seed_dir=(./experiments/comparisons/ppo/${EXP_NAME}_${sampler}/run${experiment_name}_s4/seed*)

for algo_seed_dir in "${algo_seed_dir[@]}"
do

       echo ${algo_seed_dir}
       python ./experiments/comparisons/ppo/eval.py \
              --func test_from_checkpoints \
              --restore ${algo_seed_dir} \
              --set_test_seed_as_training_eval \
              --kv_overrides task_config.done_on_out_of_bound=False

done

######## Eval Strategy 5: dynamic runs w/o CVaR ########
for seed in "${seeds[@]}" 
do

       echo "Training in default config with seed:" ${seed} " and outputting dir:" ${OUTPUT_DIR}
       python ./experiments/comparisons/ppo/ppo_experiment.py \
              --overrides \
              ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
              ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
              --output_dir $OUTPUT_DIR \
              --tag run${experiment_name}_s5 \
              --opt_hps ./experiments/comparisons/ppo/hpo/hpo_strategy_study_TPESampler/run1_s5/seed8_Aug-02-13-06-09_14ae2aa/hpo/hyperparameters_136.2288.yaml \
              --task cartpole --seed $seed --use_gpu True

done

algo_seed_dir=(./experiments/comparisons/ppo/${EXP_NAME}_${sampler}/run${experiment_name}_s5/seed*)

for algo_seed_dir in "${algo_seed_dir[@]}"
do

       echo ${algo_seed_dir}
       python ./experiments/comparisons/ppo/eval.py \
              --func test_from_checkpoints \
              --restore ${algo_seed_dir} \
              --set_test_seed_as_training_eval \
              --kv_overrides task_config.done_on_out_of_bound=False

done