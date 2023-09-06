#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1 # hostx or host0
sampler=$2 # RandomSampler or TPESampler
algo=$3 # ppo, sac, or ddpg
sys=$4 # cartpole, or quadrotor
task=$5 # stab, or track
resume=$6 # True or False

hpo_runs=(1 2 3)
# run HPO on with different strategy
for run in "${hpo_runs[@]}"
do

bash experiments/comparisons/rl/rl_hpo_strategy.sh ${run} $((run+6)) ${sampler} ${localOrHost} ${algo} ${sys} ${task} ${resume}

done

# eval the strategy
bash experiments/comparisons/rl/rl_hpo_strategy_eval.sh ${localOrHost} ${sampler} ${algo} ${sys} ${task}