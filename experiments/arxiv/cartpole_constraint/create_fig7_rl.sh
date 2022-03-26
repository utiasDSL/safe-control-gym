#!/bin/bash

# check if folder data exists
if [ -d "./data/" ] 
then
    echo "Directory ./data/ already exists." 
else
    # otherwise unzip
    unzip cartpole_constraint_data.zip
fi

TRAIN_SCRIPT="../../main.py"
EVAL_SCRIPT="../cartpole_performance/utils/eval.py"
OUTPUT_DIR="data"
EXP_NAME="constraint_rl"

# plot learning curves
python ${EVAL_SCRIPT} --func plot_constraint --plot_dir ${OUTPUT_DIR}/${EXP_NAME}