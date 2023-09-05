#!/bin/bash

cd ~/safe-control-gym

hpo_runs=(1 2 3)
# run HPO on with different strategy
for run in "${hpo_runs[@]}"
do

bash experiments/comparisons/gpmpc/gpmpc_hpo_strategy.sh ${run} $((run)) TPESampler hostx cartpole stab

done

# eval
bash experiments/comparisons/gpmpc/gpmpc_hpo_strategy_eval.sh hostx TPESampler cartpole stab