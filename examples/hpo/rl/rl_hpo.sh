#!/bin/bash

######## NOTE ########
# This script is used to run HPO in parallel.
# 1. Adjust hpo config.
# 2. Remove or backup the database if needed.
# 3. Create a screen session `screen`, and detach it `Ctrl+a d`.
# 4. Run this script by giving experiment name as the first arg. and the seed as the second.
# 5. If you want to kill them, run `pkill -f "python ./experiments/comparisons/gpmpc/gpmpc_experiment.py"`. 
#####################

cd ~/safe-control-gym

experiment_name=$1
seed1=$2
seed2=$((seed1+100))
seed3=$((seed1+200))
seed4=$((seed1+300))
sampler=$3 # RandomSampler or TPESampler
localOrHost=$4
sys=$5 # cartpole, or quadrotor
task=$6 # stab, or track
algo=$7 # ppo or sac
resume=$8 # True or False


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

# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ${algo}_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ${algo}_hpo

# if resume is False, create a study
if [ "$resume" == 'False' ]; then

python ./examples/hpo/hpo_experiment.py \
         --overrides ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
                     ./examples/hpo/rl/config_overrides/${sys}/${sys}_${task}.yaml \
                     ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_.yaml \
                     --output_dir ./examples/hpo/rl/${algo}/hpo_study_${sampler}_${sys} \
                     --sampler $sampler \
                     --use_gpu True \
                     --task ${sys} --func hpo --tag run${experiment_name} --seed $seed1 &
pid1=$!

# wait until the first study is created
sleep 3

# set load_study to True
python ./examples/hpo/hpo_experiment.py \
         --overrides ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
                     ./examples/hpo/rl/config_overrides/${sys}/${sys}_${task}.yaml \
                     ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_.yaml \
                     --output_dir ./examples/hpo/rl/${algo}/hpo_study_${sampler}_${sys} \
                     --sampler $sampler \
                     --use_gpu True \
                     --task ${sys} --func hpo --load_study True --tag run${experiment_name} --seed $seed2 &
pid2=$!

fi
# if resume is True, load the study
if [ "$resume" == 'True' ]; then

cd ./examples/hpo/rl/${algo}/hpo_study_${sampler}_${sys}/run${experiment_name}
mysql -u optuna ${algo}_hpo < ${algo}_hpo.sql

cd ~/safe-control-gym

# set load_study to True
python ./examples/hpo/hpo_experiment.py \
         --overrides ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
                     ./examples/hpo/rl/config_overrides/${sys}/${sys}_${task}.yaml \
                     ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_.yaml \
                     --output_dir ./examples/hpo/rl/${algo}/hpo_study_${sampler}_${sys} \
                     --sampler $sampler \
                     --use_gpu True \
                     --task ${sys} --func hpo --load_study True --tag run${experiment_name} --seed $seed3 &
pid1=$!

# set load_study to True
python ./examples/hpo/hpo_experiment.py \
         --overrides ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_.yaml \
                     ./examples/hpo/rl/config_overrides/${sys}/${sys}_${task}.yaml \
                     ./examples/hpo/rl/${algo}/config_overrides/${sys}/${algo}_${sys}_hpo_.yaml \
                     --output_dir ./examples/hpo/rl/${algo}/hpo_study_${sampler}_${sys} \
                     --sampler $sampler \
                     --use_gpu True \
                     --task ${sys} --func hpo --load_study True --tag run${experiment_name} --seed $seed3 &
pid2=$!

fi

# move the database from . into output_dir after both commands finish
wait $pid1
echo "job1 finished"
wait $pid2
echo "job2 finished"

# old code for sqlite database which having performance issue
# mv ${algo}_hpo.db ./experiments/comparisons/gpmpc/hpo/${experiment_name}/${algo}_hpo.db

# new code for mysql database
# back up first
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ${algo}_hpo > ${algo}_hpo.sql
mv ${algo}_hpo.sql ./examples/hpo/rl/${algo}/hpo_study_${sampler}_${sys}/run${experiment_name}/${algo}_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ${algo}_hpo