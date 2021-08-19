#!/bin/bash
# proper cleanup for background processes
trap "exit" INT TERM ERR
trap "kill 0" EXIT

rm -r -f ./safe_exp_results/

OUTPUT_DIR="./"
TAG_ROOT="safe_exp_results"
CONFIG_PATH_ROOT="./config_overrides"

seeds=(2 22 222 2222 22222 9 90 998 9999 90001)
thread=1

########################## ppo 
TAG="ppo"
CONFIG_PATH="${CONFIG_PATH_ROOT}/ppo_cartpole.yaml"

for seed in "${seeds[@]}"
do
    python3 ../main.py --algo ppo --task cartpole --overrides $CONFIG_PATH --output_dir ${OUTPUT_DIR} --tag $TAG_ROOT/$TAG --thread $thread --seed $seed
done 

########################## ppo + reward shaping 
TAG="ppo_rs"
CONFIG_PATH="${CONFIG_PATH_ROOT}/ppo_rs_cartpole.yaml"

tolerances=(0.15 0.2)
for tolerance in "${tolerances[@]}"
do
    for seed in "${seeds[@]}"
    do
        python3 ../main.py --algo ppo --task cartpole --overrides $CONFIG_PATH --output_dir ${OUTPUT_DIR} --tag $TAG_ROOT/${TAG}_${tolerance} --kv_overrides task_config.constraints.abs_bound.tolerance=$tolerance --thread $thread --seed $seed
    done 
done

########################## pretrain safe explorer
TAG="safe_exp_pretrain"
CONFIG_PATH="${CONFIG_PATH_ROOT}/safe_explorer_ppo_cartpole_pretrain.yaml"

train_seed=88890
python3 ../main.py --algo safe_explorer_ppo --task cartpole --overrides $CONFIG_PATH --output_dir ${OUTPUT_DIR} --tag $TAG_ROOT/$TAG --thread $thread --seed $train_seed

########################## train safe explorer
PRETRAINED_PATH=(${OUTPUT_DIR}/$TAG_ROOT/$TAG/seed${train_seed}*)

TAG="safe_exp_"
CONFIG_PATH="${CONFIG_PATH_ROOT}/safe_explorer_ppo_cartpole.yaml"

slacks=(0.15 0.2)
for slack in "${slacks[@]}"
do
    for seed in "${seeds[@]}"
    do
        python3 ../main.py --algo safe_explorer_ppo --task cartpole --overrides $CONFIG_PATH --output_dir ${OUTPUT_DIR} --tag $TAG_ROOT/${TAG}slack${slack} --kv_overrides algo_config.pretrained=$PRETRAINED_PATH algo_config.constraint_slack=$slack --thread $thread --seed $seed
    done 
done
