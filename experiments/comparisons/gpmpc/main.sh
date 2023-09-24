#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1 # hostx or host0
sampler=$2 # RandomSampler or TPESampler
sys=$3 # cartpole, or quadrotor
task=$4 # stab, or track
resume=$5 # True or False

hpo_runs=(1 2 3)
# run HPO on with different strategy
for run in "${hpo_runs[@]}"
do

bash experiments/comparisons/gpmpc/gpmpc_hpo_strategy.sh ${run} $((run)) ${sampler} ${localOrHost} ${sys} ${task} ${resume}

done

# eval
bash experiments/comparisons/gpmpc/gpmpc_hpo_strategy_eval.sh ${localOrHost} ${sampler} ${sys} ${task}