#!/bin/bash

# Remove previous results.
rm -r -f ./temp_data/
rm -r -f ./data/

# train models and save checkpoints 
bash utils/rl_quad.sh ppo train 
bash utils/rl_quad.sh sac train 

# evaluate performance across training from checkpoints
# it's done separately since the benchmark eval conditions are slightly different than the eval conditions during training.
bash utils/rl_quad.sh ppo post_evaluate "seed*"
bash utils/rl_quad.sh sac post_evaluate "seed*"

# move the results from temp_data/ into data/
rm -r -f ./data/
mv ./temp_data/ ./data/
