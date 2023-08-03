#!/bin/bash

######## NOTE ########
# This script is used to run HPO in parallel with different strategy.
# Set up MySQL database for the first time you run (add user and the database with all grant).
# 01. sudo apt-get install mysql-server
# 02. sudo mysql
# 03. CREATE USER optuna@"%";
# 04  CREATE DATABASE ppo_hpo;
# 05. GRANT ALL ON ppo_hpo.* TO optuna@"%";
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
seed2=$((seed1+1))
seed3=$((seed1+2))
seed4=$((seed1+3))
sampler=$3 # RandomSampler or TPESampler
localOrHost=$4

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host2' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

conda activate pr-env

######## Strategy 1: naive single run ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ppo_hpo

python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_1.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s1 --seed $seed1 --use_gpu True &
pid1=$!

# wait until the first study is created
sleep 2

# set load_study to True
python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_1.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s1 --seed $seed2 --load_study True --use_gpu True &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
wait $pid2
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ppo_hpo > ppo_hpo.sql
mv ppo_hpo.sql ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler}/run${experiment_name}_s1/ppo_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
echo "Strategy 1 done"

######## Strategy 2: naive multiple runs ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ppo_hpo

python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_2.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s2 --seed $seed1 --use_gpu True &
pid1=$!

# wait until the first study is created
sleep 2

# set load_study to True
python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_2.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s2 --seed $seed2 --load_study True --use_gpu True &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
wait $pid2
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ppo_hpo > ppo_hpo.sql
mv ppo_hpo.sql ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler}/run${experiment_name}_s2/ppo_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
echo "Strategy 2 done"

######## Strategy 3: multiple runs w/ CVaR ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ppo_hpo

python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_3.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s3 --seed $seed1 --use_gpu True &
pid1=$!

# wait until the first study is created
sleep 2

# set load_study to True
python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_3.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s3 --seed $seed2 --load_study True --use_gpu True &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
wait $pid2
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ppo_hpo > ppo_hpo.sql
mv ppo_hpo.sql ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler}/run${experiment_name}_s3/ppo_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
echo "Strategy 3 done"

######## Strategy 4: dynamic runs w/ CVaR ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ppo_hpo

python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_4.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s4 --seed $seed1 --use_gpu True &
pid1=$!

# wait until the first study is created
sleep 2

# set load_study to True
python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_4.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s4 --seed $seed2 --load_study True --use_gpu True &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
wait $pid2
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ppo_hpo > ppo_hpo.sql
mv ppo_hpo.sql ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler}/run${experiment_name}_s4/ppo_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
echo "Strategy 4 done"

######## Strategy 5: dynamic runs w/o CVaR ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ppo_hpo

python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_5.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s5 --seed $seed1 --use_gpu True &
pid1=$!

# wait until the first study is created
sleep 2

# set load_study to True
python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo_5.yaml \
            --output_dir ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler} \
            --sampler $sampler \
            --task cartpole --func hpo --tag run${experiment_name}_s5 --seed $seed2 --load_study True --use_gpu True &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
wait $pid2
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ppo_hpo > ppo_hpo.sql
mv ppo_hpo.sql ./experiments/comparisons/ppo/hpo/hpo_strategy_study_${sampler}/run${experiment_name}_s5/ppo_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ppo_hpo
echo "Strategy 5 done"