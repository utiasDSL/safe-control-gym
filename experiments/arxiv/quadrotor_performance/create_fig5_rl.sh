#!/bin/bash

# check if folder data exists
if [ -d "./data/" ] 
then
    echo "Directory ./data/ already exists." 
else
    # otherwise unzip
    unzip quadrotor_performance_data.zip
fi

TRAIN_SCRIPT="../../main.py"
EVAL_SCRIPT="utils/eval.py"
OUTPUT_DIR="data"
EXP_NAME="performance_rl"

# evaluate final performance & save test episodes
trained_ppo_path=(${OUTPUT_DIR}/${EXP_NAME}/ppo/seed*)
trained_sac_path=(${OUTPUT_DIR}/${EXP_NAME}/sac/seed*)

# pick any seed run to evaluate
python ${EVAL_SCRIPT} --func test_policy --kv_overrides algo_config.eval_batch_size=3 task_config.done_on_out_of_bound=False --set_test_seed --seed 998 --restore ${trained_ppo_path[0]}

python ${EVAL_SCRIPT} --func test_policy --kv_overrides algo_config.eval_batch_size=3 task_config.done_on_out_of_bound=False --set_test_seed --seed 998 --restore ${trained_sac_path[0]}