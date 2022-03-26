#!/bin/bash

# check if folder data exists
if [ -d "./data/" ] 
then
    echo "Directory ./data/ already exists." 
else
    # otherwise unzip
    unzip quadrotor_constraint_data.zip
fi

TRAIN_SCRIPT="../../main.py"
EVAL_SCRIPT="../quadrotor_performance/utils/eval.py"
OUTPUT_DIR="data"
EXP_NAME="constraint_rl"

# evaluate constraint satisfaction performance & save test episodes
seppo_tag="slack0.02"
seppo_seed=65
trained_seppo_path=(${OUTPUT_DIR}/${EXP_NAME}/safe_explorer_ppo/${seppo_tag}/seed${seppo_seed}_*)

python ${EVAL_SCRIPT} --func test_policy --kv_overrides algo_config.eval_batch_size=3 task_config.done_on_out_of_bound=False --set_test_seed --seed 998 --restore ${trained_seppo_path}
