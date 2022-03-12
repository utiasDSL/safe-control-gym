#!/bin/bash
####################### NOTE 
# This script should be executed from a local machine,
# with the virtual environment activated.
# 
# no indentation to keep the good formatting of `cmd` strings 
# 
# use the 1st argument as the algorithm/controller name
# use the 2nd argument to the script as function name to be executed.
# 
#######################

# proper cleanup for background processes
trap "exit" INT TERM ERR
trap "kill 0" EXIT

# experiment info 
ALGO_NAME=$1
func_name=$2
args=${@:3}

TRAIN_SCRIPT="../../main.py"
EVAL_SCRIPT="../quadrotor_performance/utils/eval.py"
OUTPUT_DIR="temp_data"
EXP_NAME="constraint_rl"

CONFIG_DIR="config_overrides"
BASE_CONFIG_PATH="${CONFIG_DIR}/base.yaml"
CONFIG_PATH="${CONFIG_DIR}/${ALGO_NAME}_quad.yaml"

# experiment parameters
seeds=(6 65 668)
thread_num=1


####################### ppo 

function train_ppo {    
for seed in "${seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task quadrotor \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd &
done 

wait
}


####################### safe explorer ppo 

function pretrain_seppo {    
algo_dir=$1
train_seeds=(8)
CONFIG_PATH="${CONFIG_DIR}/${ALGO_NAME}_quad_pretrain.yaml"

for seed in "${train_seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}/pretrain/${algo_dir}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task quadrotor \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd &
done 

wait
}


function train_seppo {    
algo_dir=$1
pretrain_prefix=$2
pretrain_path=(${OUTPUT_DIR}/${EXP_NAME}/${ALGO_NAME}/pretrain/${pretrain_prefix}*)

for seed in "${seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}/${algo_dir}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task quadrotor \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
--kv_overrides \
algo_config.pretrained=${pretrain_path} \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd &
done

wait
}


####################### post-training evaluation  

function post_evaluate {   
algo_dir=$1
test_seed=998

# get all seed dirs of the algo 
algo_seed_dirs=(${OUTPUT_DIR}/${EXP_NAME}/${ALGO_NAME}/${algo_dir})
    
# re-run evalution on algo checkpoints 
for algo_seed_dir in "${algo_seed_dirs[@]}" 
do
# construct commnad 
cmd="python ${EVAL_SCRIPT} \
--func test_from_checkpoints \
--restore ${algo_seed_dir} \
--set_test_seed \
--seed ${test_seed} \
--kv_overrides \
task_config.done_on_out_of_bound=False \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd &
done 

wait
}


####################### execution 

${func_name} $args

