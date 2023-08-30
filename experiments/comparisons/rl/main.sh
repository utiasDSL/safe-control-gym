#!/bin/bash

cd ~/safe-control-gym

hpo_runs=(1 2 3 4 5)
# run HPO on with different strategy
for run in "${hpo_runs[@]}"
do

bash experiments/comparisons/rl/rl_hpo_strategy.sh ${run} $((run+6)) TPESampler hostx sac cartpole stab

done
# eval the strategy
# bash experiments/comparisons/ppo/ppo_hpo_strategy_eval.sh 1 host0 RandomSampler