#!/bin/bash

# Remove previous results.
rm -r -f ./temp_data/
rm -r -f ./data/

# train models 
bash utils/rl_cartpole_constraint.sh ppo train_ppo

seppo_pretrain_tag="dim64_lr0.0001"
seppo_tag="slack0.050.07"

bash utils/rl_cartpole_constraint.sh safe_explorer_ppo pretrain_seppo ${seppo_pretrain_tag}
bash utils/rl_cartpole_constraint.sh safe_explorer_ppo train_seppo ${seppo_tag} ${seppo_pretrain_tag}/seed8

# evaluate performance across training from checkpoints
bash utils/rl_cartpole_constraint.sh ppo post_evaluate "seed*"
bash utils/rl_cartpole_constraint.sh safe_explorer_ppo post_evaluate "${seppo_tag}/seed*"

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp_data/ ./data/
