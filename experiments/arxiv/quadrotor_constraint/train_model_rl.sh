#!/bin/bash

# Remove previous results.
rm -r -f ./temp_data/
rm -r -f ./data/

# train models 
bash utils/rl_quad_constraint.sh ppo train_ppo

seppo_pretrain_tag="dim64_lr0.0001"
seppo_tag="slack0.02"

bash utils/rl_quad_constraint.sh safe_explorer_ppo pretrain_seppo ${seppo_pretrain_tag}
bash utils/rl_quad_constraint.sh safe_explorer_ppo train_seppo ${seppo_tag} ${seppo_pretrain_tag}/seed8

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp_data/ ./data/
