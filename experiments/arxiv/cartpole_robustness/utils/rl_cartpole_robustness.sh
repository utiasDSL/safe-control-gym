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

# disabling this block, since `plot_robustness` seems to take it as 
# a termination signal between plots, using the block will stop it 
# from plotting multiple figures somehow
#
# # proper cleanup for background processes
# trap "exit" INT TERM ERR
# trap "kill 0" EXIT

# experiment info 
ALGO_NAME=$1
func_name=$2
args=${@:3}

TRAIN_SCRIPT="../../main.py"
EVAL_SCRIPT="../cartpole_performance/utils/eval.py"
OUTPUT_DIR="temp_data"
RESULTS_DIR="data"
EXP_NAME="robustness_rl"

CONFIG_DIR="config_overrides"
BASE_CONFIG_PATH="${CONFIG_DIR}/base.yaml"
CONFIG_PATH="${CONFIG_DIR}/${ALGO_NAME}_cartpole.yaml"

# experiment parameters
seeds=(6 65 668)
thread_num=1
eval_batch_size=10


####################### training

function train {    
for seed in "${seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task cartpole \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
"

# construct cluster job 
echo "Execute command: "
echo ${cmd}
$cmd #&
done 

wait
}


####################### training with domain randomization (dr)

function train_dr {  
dr_config_path="${CONFIG_DIR}/dr.yaml" 
# dr parameters (can expose as command line arguments)
# 3 possible modes: pole_length, pole_mass, cart_mass 
mode="pole_length"
low=0.1
high=1.0

# write the additional dr config 
echo "# domain randomizaiton  config 
task_config:
  randomized_inertial_prop: True 
  inertial_prop_randomization_info:
    $mode:
      distrib: uniform
      low: $low
      high: $high
" > ${dr_config_path}

for seed in "${seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}_dr_${mode}/low${low}_high${high}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task cartpole \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} ${dr_config_path} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd #&
done 

wait
}


####################### rarl 

function train_rarl {    
# parameter search 
# local adv_scales=(0.005 0.01 0.02 0.05 0.1)
local adv_scales=(0.1)

for adv_scale in "${adv_scales[@]}" 
do
for seed in "${seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}/scale${adv_scale}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task cartpole \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
--kv_overrides \
task_config.adversary_disturbance_scale=${adv_scale} \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd #&
done 

wait
done
}


####################### rap 

function train_rap {   
# parameter search  
# local adv_scales=(0.005 0.01 0.02 0.05 0.1)
local adv_scales=(0.1)

for adv_scale in "${adv_scales[@]}" 
do
for seed in "${seeds[@]}" 
do
# experiment name 
job_name="${EXP_NAME}/${ALGO_NAME}/scale${adv_scale}"

# construct commnad 
cmd="python ${TRAIN_SCRIPT} \
--algo ${ALGO_NAME} \
--task cartpole \
--overrides ${BASE_CONFIG_PATH} ${CONFIG_PATH} \
--output_dir ${OUTPUT_DIR} \
--tag ${job_name} \
--seed $seed \
--thread ${thread_num} \
--kv_overrides \
task_config.adversary_disturbance_scale=${adv_scale} \
"

# construct job 
echo "Execute command: "
echo ${cmd}
$cmd #&
done 

wait
done
}


####################### evaluate robustness w.r.t params 

function test_params {    
# string/pattern to filter the model folders to be tested
# e.g. seed*/[0-9]*/seed*, scale*/seed*/[0-9]*/seed*
algo_dir=$1
test_seed=6161
seed_dirs=(${RESULTS_DIR}/${EXP_NAME}/${ALGO_NAME}/${algo_dir})
# config file with common evaluation settings
eval_base_config_path="${CONFIG_DIR}/eval/base.yaml"

# 3 modes: pole_length, pole_mass, cart_mass 
mode="pole_length"
# params to be tested on based on mode 
if [ $mode = "pole_length" ]; then 
    params=(0.15 0.25 0.5 0.75 1.25 1.65 2.0 2.5 3)
elif [ $mode = "pole_mass" ]; then 
    params=(0.01 0.05 0.1 0.2 0.5 1.0 2.0)
elif [ $mode = "cart_mass" ]; then 
    params=(0.1 0.5 1.0 1.5 2.0 2.5)
else 
    echo "The provided mode is not valid as a tunable system parameter"
fi 

for seed_dir in "${seed_dirs[@]}"
do
# folder to hold evaluation output files 
algo_seed_dir=${seed_dir//"${RESULTS_DIR}/${EXP_NAME}/"/}
eval_output_dir="$RESULTS_DIR/${EXP_NAME}/eval/${mode}/${algo_seed_dir}"

for param in "${params[@]}"
do 
cmd="python ${EVAL_SCRIPT} \
--func test_policy_robustness \
--overrides ${eval_base_config_path} \
--restore ${seed_dir} \
--eval_output_dir ${eval_output_dir} \
--eval_output_path ${param}.pkl \
--set_test_seed \
--seed ${test_seed} \
--kv_overrides \
algo_config.eval_batch_size=${eval_batch_size} \
task_config.inertial_prop.${mode}=${param} \
task_config.done_on_out_of_bound=False \
"
${cmd}
done
done
}


####################### evaluate robustness w.r.t disturbances 

function test_disturbances {   
# string/pattern to filter the model folders to be tested 
# e.g. scale*/seed*/[0-9]*/seed*
algo_dir=$1
test_seed=616161
seed_dirs=($RESULTS_DIR/${EXP_NAME}/${ALGO_NAME}/${algo_dir})
# config file with common evaluation settings
eval_base_config_path="${CONFIG_DIR}/eval/base.yaml"

# 2 modes: act_white, act_uniform 
# only show act_white here, since the mode and evaluation procedure are slightly coupled, 
# do not have a simple solution like `test_params()` yet, likely need several if-else blocks.
mode="act_white"
# disturbance eval info
disturb_config_path="${CONFIG_DIR}/eval/${mode}.yaml"
disturbance_stds=(0.0 0.005 0.5 1.0 2.0 4.0)

for seed_dir in "${seed_dirs[@]}"
do
# folder to hold evaluation output files 
algo_seed_dir=${seed_dir//"${RESULTS_DIR}/${EXP_NAME}/"/}
eval_output_dir="$RESULTS_DIR/${EXP_NAME}/eval/${mode}/${algo_seed_dir}"

for dist_std in "${disturbance_stds[@]}"
do
echo "# disturbance config 
task_config:
  disturbances:
    action:
    - disturbance_func: white_noise
      std: ${dist_std}
" > ${disturb_config_path}

cmd="python ${EVAL_SCRIPT} \
--func test_policy_robustness \
--overrides ${eval_base_config_path} ${disturb_config_path} \
--restore ${seed_dir} \
--eval_output_dir ${eval_output_dir} \
--eval_output_path ${dist_std}.pkl \
--set_test_seed \
--seed ${test_seed} \
--kv_overrides \
algo_config.eval_batch_size=${eval_batch_size} \
task_config.done_on_out_of_bound=False \
"
${cmd}
done
done
}


####################### plot

function plot_robustness {  
# 5 modes: pole_length, pole_mass, cart_mass , act_white, act_uniform 
# only show pole_length & act_white here 
mode=$1

if [ $mode = "pole_length" ] 
then 

cmd="python ${EVAL_SCRIPT} \
--func plot_robustness \
--plot_dir ${RESULTS_DIR}/${EXP_NAME} \
--fig_name robustness_${mode}.png \
--eval_result_dir ${RESULTS_DIR}/${EXP_NAME}/eval/${mode} \
--param_name pole_length \
--trained_value 0.5 \
"

elif [ $mode = "act_white" ]
then 

cmd="python ${EVAL_SCRIPT} \
--func plot_robustness \
--plot_dir ${RESULTS_DIR}/${EXP_NAME} \
--fig_name robustness_${mode}.png \
--eval_result_dir ${RESULTS_DIR}/${EXP_NAME}/eval/${mode} \
--param_name action_white_noise \
"

else
echo "Plotting mode ${mode} not supported..."
fi 

${cmd}
}


####################### execution 

${func_name} $args
