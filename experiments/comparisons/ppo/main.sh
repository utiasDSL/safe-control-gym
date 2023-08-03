#!/bin/bash

cd ~/safe-control-gym

# run HPO with different strategy
bash experiments/comparisons/ppo/ppo_hpo_strategy.sh 1 7 TPESampler host0

# eval the strategy
# bash experiments/comparisons/ppo/ppo_hpo_strategy_eval.sh 1 host0 RandomSampler