#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1 # hostx or host0
sampler=$2 # RandomSampler or TPESampler
algo=$3 # ppo or sac
sys=$4 # cartpole, or quadrotor
task=$5 # stab, or track
resume=$6 # True or False

hpo_runs=(1)
# run HPO on with different strategy
for run in "${hpo_runs[@]}"
do

bash examples/hpo/rl/rl_hpo.sh ${run} $((run+6)) ${sampler} ${localOrHost} ${sys} ${task} ${algo} ${resume}

done

# eval
bash examples/hpo/rl/rl_hp_evaluation.sh ${localOrHost} ${algo} ${sys} ${task} ${sampler}