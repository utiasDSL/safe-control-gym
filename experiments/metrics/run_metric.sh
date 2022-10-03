#!/bin/bash

#######################

# proper cleanup for background processes
trap "exit" INT TERM ERR
trap "kill 0" EXIT

# experiment info 
func_name=$1
exp_name=$2
args=${@:3}

OUTPUT_DIR="temp_data"
TRAIN_SCRIPT="../main.py"
EVAL_SCRIPT="test_metric.py"
CONFIG_DIR="config_overrides/${exp_name}"
BASE_CONFIG_PATH="${CONFIG_DIR}/${base_path}"
CONFIG_PATH="${CONFIG_DIR}/${config_path}"

seeds=(6 65 668)
thread_num=1
eval_seed=101


#--------------------------------------------------------------------
# train an agent 

function train {    
for seed in "${seeds[@]}" 
do
cmd="python ${TRAIN_SCRIPT} \
--algo ppo \
--task cartpole_v3 \
--overrides configs/cartpole_transfer/envs/mid/base.yaml configs/cartpole_transfer/ppo.yaml \
--output_dir results \
--tag cartpole_transfer/prior/ppo \
--seed $seed \
--thread ${thread_num} \
"
echo "Execute command: "
echo ${cmd}
$cmd &
done 
wait
}


function train_expert {    
for seed in "${seeds[@]}" 
do
cmd="python ${TRAIN_SCRIPT} \
--algo ppo \
--task cartpole_v3 \
--overrides configs/cartpole_transfer/envs/mid/base_target.yaml configs/cartpole_transfer/ppo.yaml \
--output_dir results \
--tag cartpole_transfer/prior/expert \
--seed $seed \
--thread ${thread_num} \
"
echo "Execute command: "
echo ${cmd}
$cmd &
done 
wait
}


#--------------------------------------------------------------------
# # test mmd correlations 
# ppo_paths=(results/prior/ppo/seed*)
# python test_prior.py --func test_mmd --restore ${ppo_paths[0]} --seed 101

function test_mmd {  
ctrl_paths=(results/cartpole_transfer/prior/expert/seed*)

cmd="python ${EVAL_SCRIPT} \
--func test_mmd \
--restore ${ctrl_paths[0]} \
--seed ${eval_seed} \
--eval_output_dir results/cartpole_transfer/prior \
--task_config_path configs/cartpole_transfer/envs/mid/base_target.yaml \
--param damping \
--tuple_length 1 \
--n_steps 100 \
"
echo "Execute command: "
echo ${cmd}
$cmd
}


function test_mmd_length {  
ctrl_paths=(results/cartpole_transfer/prior/expert/seed*)

cmd="python ${EVAL_SCRIPT} \
--func test_mmd \
--restore ${ctrl_paths[0]} \
--seed ${eval_seed} \
--eval_output_dir results/cartpole_transfer/prior \
--task_config_path configs/cartpole_transfer/envs/mid/base_target.yaml \
--param length \
--tuple_length 1 \
--n_steps 5000 \
--fig_name pole_length.png \
--file_name pole_length.yaml \
"
echo "Execute command: "
echo ${cmd}
$cmd
}


function test_mmd_scale {  
ctrl_paths=(results/cartpole_transfer/prior/expert/seed*)

cmd="python ${EVAL_SCRIPT} \
--func test_mmd \
--restore ${ctrl_paths[0]} \
--seed ${eval_seed} \
--eval_output_dir results/cartpole_transfer/prior \
--task_config_path configs/cartpole_transfer/envs/mid/base_target.yaml \
--param scale \
--tuple_length 1 \
--n_steps 5000 \
--fig_name param_scale.png \
--file_name param_scale.yaml \
"
echo "Execute command: "
echo ${cmd}
$cmd
}


function test_mmd_damping {  
ctrl_paths=(results/cartpole_transfer/prior/expert/seed*)

cmd="python ${EVAL_SCRIPT} \
--func test_mmd \
--restore ${ctrl_paths[0]} \
--seed ${eval_seed} \
--eval_output_dir results/cartpole_transfer/prior \
--task_config_path configs/cartpole_transfer/envs/mid/base_target.yaml \
--param damping \
--tuple_length 1 \
--n_steps 5000 \
--fig_name damping.png \
--file_name damping.yaml \
"
echo "Execute command: "
echo ${cmd}
$cmd
}


#--------------------------------------------------------------------
# mmd ablations

function ablate_mmd_tuple_length {  
ctrl_paths=(results/cartpole_transfer/prior/expert/seed*)
tuple_lengths=(1 3 5 7 9)

for tlength in "${tuple_lengths[@]}" 
do
cmd="python ${EVAL_SCRIPT} \
--func test_mmd \
--restore ${ctrl_paths[0]} \
--seed ${eval_seed} \
--eval_output_dir ${OUTPUT_DIR}/prior \
--task_config_path configs/cartpole_transfer/envs/mid/base_target.yaml \
--param length \
--tuple_length ${tlength} \
--n_steps 1000 \
--fig_name damping.png \
--file_name damping.yaml \
"
echo "Execute command: "
echo ${cmd}
$cmd &
done
wait
}


function ablate_mmd_sigma {  
ctrl_paths=(results/cartpole_transfer/prior/expert/seed*)
mmd_sigmas=(0.1 1 10 100 500)

for sigma in "${mmd_sigmas[@]}" 
do
cmd="python ${EVAL_SCRIPT} \
--func test_mmd \
--restore ${ctrl_paths[0]} \
--seed ${eval_seed} \
--eval_output_dir ${OUTPUT_DIR}/prior \
--task_config_path configs/cartpole_transfer/envs/mid/base_target.yaml \
--param length \
--mmd_sigma ${sigma} \
--n_steps 1000 \
--fig_name damping.png \
--file_name damping.yaml \
"
echo "Execute command: "
echo ${cmd}
$cmd &
done
wait
}




#####################################################################
#####################################################################
#####################################################################

n_episodes=10
test_metric_seeds=(6 65 668 6673 61234)


function test_metric {
# 
algo=$1
env=$2
metric=$3
sub_exp_name=$4 
if [ ${sub_exp_name} = "-" ]; then 
    sub_exp_name="${metric}"
fi 
kwargs=${@:5}

for seed in "${test_metric_seeds[@]}" 
do
cmd="python ${EVAL_SCRIPT} \
--func test_trajectory_metric \
--algo ${algo} \
--task ${env} \ 
--overrides ${CONFIG_DIR}/base.yaml ${CONFIG_DIR}/variant.yaml \
--n_episodes ${n_episodes} \
--metric ${metric} \
--eval_output_dir ${OUTPUT_DIR}/${exp_name}/${sub_exp_name}/seed${seed} \
--seed ${seed} \
${kwargs} \
"
echo "Execute command: "
echo ${cmd}
$cmd &
done 
wait
}

function test_metric_nonparallel {
# 
algo=$1
env=$2
metric=$3
sub_exp_name=$4 
if [ ${sub_exp_name} = "-" ]; then 
    sub_exp_name="${metric}"
fi
kwargs=${@:5}

for seed in "${test_metric_seeds[@]}" 
do
cmd="python ${EVAL_SCRIPT} \
--func test_trajectory_metric \
--algo ${algo} \
--task ${env} \
--overrides ${CONFIG_DIR}/base.yaml ${CONFIG_DIR}/variant.yaml \
--n_episodes ${n_episodes} \
--metric ${metric} \
--eval_output_dir ${OUTPUT_DIR}/${exp_name}/${sub_exp_name}/seed${seed} \
--seed ${seed} \
${kwargs} \
"
echo "Execute command: "
echo ${cmd}
$cmd
done 
}

#####################################################################

function test_metric_gt {
# 
algo=$1
env=$2
metric=$3
sub_exp_name=$4 
if [ ${sub_exp_name} = "-" ]; then 
    sub_exp_name="${metric}/${algo}"
fi 
algo_config=$5
if [ ${algo_config} = "-" ]; then 
    algo_config="${algo}"
fi 
kwargs=${@:6}

for seed in "${test_metric_seeds[@]}" 
do
cmd="python ${EVAL_SCRIPT} \
--func test_metric_to_gt \
--algo ${algo} \
--task ${env} \
--overrides ${CONFIG_DIR}/base.yaml ${CONFIG_DIR}/${algo_config} \
--n_episodes ${n_episodes} \
--metric ${metric} \
--eval_output_dir ${OUTPUT_DIR}/${exp_name}/${sub_exp_name}/seed${seed} \
--seed ${seed} \
${kwargs} \
"
echo "Execute command: "
echo ${cmd}
$cmd &
done 
wait
}

function test_metric_gt_nonparallel {
# 
algo=$1
env=$2
metric=$3
sub_exp_name=$4 
if [ ${sub_exp_name} = "-" ]; then 
    sub_exp_name="${metric}/${algo}"
fi 
algo_config=$5
if [ ${algo_config} = "-" ]; then 
    algo_config="${algo}.yaml"
fi 
kwargs=${@:6}

for seed in "${test_metric_seeds[@]}" 
do
cmd="python ${EVAL_SCRIPT} \
--func test_metric_to_gt \
--algo ${algo} \
--task ${env} \
--overrides ${CONFIG_DIR}/base.yaml ${CONFIG_DIR}/${algo_config} \
--n_episodes ${n_episodes} \
--metric ${metric} \
--eval_output_dir ${OUTPUT_DIR}/${exp_name}/${sub_exp_name}/seed${seed} \
--seed ${seed} \
${kwargs} \
"
echo "Execute command: "
echo ${cmd}
$cmd
done 
}


#####################################################################
#####################################################################
#####################################################################


####################### execution 

${func_name} $args 


# bash run.sh train
# bash run.sh train_expert
# (for debug) bash run.sh test_mmd
# bash run.sh test_mmd_length
# bash run.sh test_mmd_scale
# bash run.sh test_mmd_damping
