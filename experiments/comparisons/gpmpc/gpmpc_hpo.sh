#!/bin/bash

######## NOTE ########
# Set up MySQL database for the first time you run (add user and the database with all grant).
# 01. sudo apt-get install mysql-server
# 02. sudo mysql
# 03. CREATE USER optuna@"%";
# 04  CREATE DATABASE gp_mpc_hpo;
# 05. GRANT ALL ON gp_mpc_hpo.* TO optuna@"%";
# 06. exit
# This script is used to run HPO in parallel. Two processes will be faster for HPO of GP-MPC.
# 1. Adjust hpo config.
# 2. Remove or backup the database if needed.
# 3. Create a screen session `screen`, and detach it `Ctrl+a d`.
# 4. Run this script by giving experiment name as the first arg. and the seed as the second.
# 5. If you want to kill them, run `pkill -f "python ./experiments/comparisons/gpmpc/gpmpc_experiment.py"`. 
#####################

cd ~/safe-control-gym

experiment_name=$1
seed1=$2
seed2=$((seed1+1))
seed3=$((seed1+2))
seed4=$((seed1+3))
localOrHost=$3

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
elif [ "$localOrHost" == 'host4' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

conda activate safe

######## PRIOR 150 ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag gp_mpc_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag gp_mpc_hpo

python ./experiments/comparisons/gpmpc/gpmpc_experiment.py \
         --overrides ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_150.yaml \
                     ./experiments/comparisons/gpmpc/config_overrides/cartpole/cartpole_stab.yaml \
                     ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_hpo.yaml \
                     --output_dir ./experiments/comparisons/gpmpc/hpo/prior \
                     --task cartpole --func hpo --tag prior_150 --seed $seed1 &
pid1=$!

# wait until the first study is created
sleep 3

# set load_study to True
python ./experiments/comparisons/gpmpc/gpmpc_experiment.py \
         --overrides ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_150.yaml \
                     ./experiments/comparisons/gpmpc/config_overrides/cartpole/cartpole_stab.yaml \
                     ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_hpo.yaml \
                     --output_dir ./experiments/comparisons/gpmpc/hpo/prior \
                     --task cartpole --func hpo --load_study True --tag prior_150 --seed $seed2 &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
echo "job1 finished"
wait $pid2
echo "job2 finished"
echo "Prior 150 HPO done"

# old code for sqlite database which having performance issue
# mv gp_mpc_hpo.db ./experiments/comparisons/gpmpc/hpo/${experiment_name}/gp_mpc_hpo.db

# new code for mysql database
# back up first
echo "backing up the database"
mysqldump --no-tablespaces -u optuna gp_mpc_hpo > gp_mpc_hpo.sql
mv gp_mpc_hpo.sql ./experiments/comparisons/gpmpc/hpo/prior/prior_150/gp_mpc_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag gp_mpc_hpo

######## PRIOR 200 ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag gp_mpc_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag gp_mpc_hpo

python ./experiments/comparisons/gpmpc/gpmpc_experiment.py \
        --overrides ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_200.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/cartpole_stab.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_hpo.yaml \
                    --output_dir ./experiments/comparisons/gpmpc/hpo/prior \
                    --task cartpole --func hpo --tag prior_200 --seed $seed1 &
pid1=$!

# wait until the first study is created
sleep 3

# set load_study to True
python ./experiments/comparisons/gpmpc/gpmpc_experiment.py \
        --overrides ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_200.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/cartpole_stab.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_hpo.yaml \
                    --output_dir ./experiments/comparisons/gpmpc/hpo/prior \
                    --task cartpole --func hpo --load_study True --tag prior_200 --seed $seed2 &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
echo "job1 finished"
wait $pid2
echo "job2 finished"
echo "Prior 200 HPO done"

# old code for sqlite database which having performance issue
# mv gp_mpc_hpo.db ./experiments/comparisons/gpmpc/hpo/${experiment_name}/gp_mpc_hpo.db

# new code for mysql database
# back up first
echo "backing up the database"
mysqldump --no-tablespaces -u optuna gp_mpc_hpo > gp_mpc_hpo.sql
mv gp_mpc_hpo.sql ./experiments/comparisons/gpmpc/hpo/prior/prior_200/gp_mpc_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag gp_mpc_hpo

######## PRIOR 250 ########
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag gp_mpc_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag gp_mpc_hpo

python ./experiments/comparisons/gpmpc/gpmpc_experiment.py \
        --overrides ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_250.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/cartpole_stab.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_hpo.yaml \
                    --output_dir ./experiments/comparisons/gpmpc/hpo/prior \
                    --task cartpole --func hpo --tag prior_250 --seed $seed1 &
pid1=$!

# wait until the first study is created
sleep 3

# set load_study to True
python ./experiments/comparisons/gpmpc/gpmpc_experiment.py \
        --overrides ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_250.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/cartpole_stab.yaml \
                    ./experiments/comparisons/gpmpc/config_overrides/cartpole/gp_mpc_cartpole_hpo.yaml \
                    --output_dir ./experiments/comparisons/gpmpc/hpo/prior \
                    --task cartpole --func hpo --load_study True --tag prior_250 --seed $seed2 &
pid2=$!

# move the database from . into output_dir after both commands finish
wait $pid1
echo "job1 finished"
wait $pid2
echo "job2 finished"
echo "Prior 250 HPO done"

# old code for sqlite database which having performance issue
# mv gp_mpc_hpo.db ./experiments/comparisons/gpmpc/hpo/${experiment_name}/gp_mpc_hpo.db

# new code for mysql database
# back up first
echo "backing up the database"
mysqldump --no-tablespaces -u optuna gp_mpc_hpo > gp_mpc_hpo.sql
mv gp_mpc_hpo.sql ./experiments/comparisons/gpmpc/hpo/prior/prior_250/gp_mpc_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag gp_mpc_hpo

