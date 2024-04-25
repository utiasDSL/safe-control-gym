#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1 # hostx or host0
sampler=$2 # RandomSampler or TPESampler
sys=$3 # cartpole, or quadrotor
task=$4 # stab, or track
resume=$5 # True or False

hpo_runs=(1)
# run HPO
for run in "${hpo_runs[@]}"
do

bash examples/hpo/gp_mpc/gp_mpc_hpo.sh ${run} $((run)) ${sampler} ${localOrHost} ${sys} ${task} ${resume}

done

# TODO: eval
bash examples/hpo/gp_mpc/gp_mpc_hp_evaluation.sh ${localOrHost} ${sys} ${task} ${sampler}