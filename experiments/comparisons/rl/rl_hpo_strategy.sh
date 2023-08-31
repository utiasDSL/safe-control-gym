#!/bin/bash

######## NOTE ########
# This script is used to run HPO in parallel with different strategy.
# Set up MySQL database for the first time you run (add user and the database with all grant).
# 01. sudo apt-get install mysql-server
# 02. sudo mysql
# 03. CREATE USER optuna@"%";
# 04  CREATE DATABASE ${algo}_hpo;
# 05. GRANT ALL ON ${algo}_hpo.* TO optuna@"%";
# 06. exit
# 1. Adjust hpo config.
# 2. Optionally change eval_interval, num_checkpoints, and log_interval to save some time.
# 3. Remove or backup the database if needed.
# 4. Create a screen session `screen`, and detach it `Ctrl+a d`.
# 5. Run this script by giving experiment name as the first arg. and the seed as the second.
# 6. If you want to kill them, run `pkill -f "python ./experiments/comparisons/ppo/ppo_experiment.py"`. 
#####################

cd ~/safe-control-gym

experiment_name=$1
seed1=$2
seed2=$((seed1+100))
seed3=$((seed1+200))
seed4=$((seed1+300))
sampler=$3 # RandomSampler or TPESampler
localOrHost=$4
algo=$5 # ppo, sac, or ddpg
sys=$6 # cartpole, or quadrotor
task=$7 # stab, or track


# Strategy 1: naive single run
# Strategy 2: naive multiple runs
# Strategy 3: multiple runs w/ CVaR
# Strategy 4: dynamic runs w/ CVaR
# Strategy 5: dynamic runs w/o CVaR
strategies=(1 2 3 4 5)

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'hostx' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

conda activate pr-env


for strategy in "${strategies[@]}"
do

# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ${algo}_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ${algo}_hpo

python ./experiments/comparisons/rl/rl_experiment.py \
            --algo ${algo} \
            --overrides \
            ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
            ./experiments/comparisons/rl/config_overrides/${sys}/${sys}_${task}.yaml \
            ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_${strategy}.yaml \
            --output_dir ./experiments/comparisons/rl/${algo}/hpo/hpo_strategy_study_${sampler}_${sys} \
            --sampler $sampler \
            --task ${sys} --func hpo --tag run${experiment_name}_s${strategy} --seed $seed1 --use_gpu True &
pid1=$!

# wait until the first study is created
sleep 2

# set load_study to True
python ./experiments/comparisons/rl/rl_experiment.py \
            --algo ${algo} \
            --overrides \
            ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
            ./experiments/comparisons/rl/config_overrides/${sys}/${sys}_${task}.yaml \
            ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_${strategy}.yaml \
            --output_dir ./experiments/comparisons/rl/${algo}/hpo/hpo_strategy_study_${sampler}_${sys} \
            --sampler $sampler \
            --task ${sys} --func hpo --tag run${experiment_name}_s${strategy} --seed $seed2 --load_study True --use_gpu True &
pid2=$!

# # set load_study to True
# python ./experiments/comparisons/rl/rl_experiment.py \
#             --algo ${algo} \
#             --overrides \
#             ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
#             ./experiments/comparisons/rl/config_overrides/${sys}/${sys}_${task}.yaml \
#             ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_${strategy}.yaml \
#             --output_dir ./experiments/comparisons/rl/${algo}/hpo/hpo_strategy_study_${sampler}_${sys} \
#             --sampler $sampler \
#             --task ${sys} --func hpo --tag run${experiment_name}_s${strategy} --seed $seed3 --load_study True --use_gpu True &
# pid3=$!

# # set load_study to True
# python ./experiments/comparisons/rl/rl_experiment.py \
#             --algo ${algo} \
#             --overrides \
#             ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
#             ./experiments/comparisons/rl/config_overrides/${sys}/${sys}_${task}.yaml \
#             ./experiments/comparisons/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_${strategy}.yaml \
#             --output_dir ./experiments/comparisons/rl/${algo}/hpo/hpo_strategy_study_${sampler}_${sys} \
#             --sampler $sampler \
#             --task ${sys} --func hpo --tag run${experiment_name}_s${strategy} --seed $seed4 --load_study True --use_gpu True &
# pid4=$!

# move the database from . into output_dir after both commands finish
wait $pid1
wait $pid2
# wait $pid3
# wait $pid4
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ${algo}_hpo > ${algo}_hpo.sql
mv ${algo}_hpo.sql ./experiments/comparisons/rl/${algo}/hpo/hpo_strategy_study_${sampler}_${sys}/run${experiment_name}_s${strategy}/${algo}_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ${algo}_hpo
echo "Strategy ${strategy} done"

done