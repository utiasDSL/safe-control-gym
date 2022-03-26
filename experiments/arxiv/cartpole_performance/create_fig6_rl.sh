#!/bin/bash

# check if folder data exists
if [ -d "./data/" ] 
then
    echo "Directory ./data/ already exists." 
else
    # otherwise unzip
    unzip cartpole_performance_data.zip
fi

TRAIN_SCRIPT="../../main.py"
EVAL_SCRIPT="utils/eval.py"
OUTPUT_DIR="data"
EXP_NAME="performance_rl"

# plot the training curve with benchmark eval conditions
python ${EVAL_SCRIPT} --func plot_performance --plot_dir ${OUTPUT_DIR}/${EXP_NAME}
