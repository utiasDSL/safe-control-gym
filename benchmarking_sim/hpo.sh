#!/bin/bash

######## NOTE ########
# This script is used to run HPO in parallel.
# 1. Adjust hpo config.
# 2. Remove or backup the database if needed.
# 3. Create a screen session `screen`, and detach it `Ctrl+a d`.
# 4. Run this script by giving experiment name as the first arg. and the seed as the second.
# 5. If you want to kill them, run `pkill -f "python ./.py"`. 
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
algo=$6
prior=$7
safety_filter=$8 # True or False
task=$9 # stab, or track
resume=${10} # True or False


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

conda activate safe

# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ${algo}_hpo
# create database
python ./safe_control_gym/hyperparameters/database.py --func create --tag ${algo}_hpo

# if resume is False, create a study
if [ "$resume" == 'False' ]; then

    if [ "$safety_filter" == 'False' ]; then

        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./benchmarking_sim/${sys}/config_overrides/${sys}_${task}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_hpo.yaml \
                            --output_dir ./benchmarking_sim/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys} --func hpo --tag ${experiment_name} --seed $seed1 &
        pid1=$!

        # wait until the first study is created
        sleep 3

        # set load_study to True
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./benchmarking_sim/${sys}/config_overrides/${sys}_${task}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_hpo.yaml \
                            --output_dir ./benchmarking_sim/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys} --func hpo --load_study True --tag ${experiment_name} --seed $seed2 &
        pid2=$!

    fi


    if [ "$safety_filter" == 'True' ]; then

        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./benchmarking_sim/${sys}/config_overrides/${sys}_${task}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_hpo.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/linear_mpsc_${sys}_${task}_${prior}.yaml \
                            --kv_overrides sf_config.cost_function=one_step_cost \
                            --output_dir ./benchmarking_sim/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys} --func hpo --tag ${experiment_name} --seed $seed1 &
        pid1=$!

        # wait until the first study is created
        sleep 3

        # set load_study to True
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./benchmarking_sim/${sys}/config_overrides/${sys}_${task}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_hpo.yaml \
                                        ./benchmarking_sim/${sys}/config_overrides/linear_mpsc_${sys}_${task}_${prior}.yaml \
                            --kv_overrides sf_config.cost_function=one_step_cost \
                            --output_dir ./benchmarking_sim/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys} --func hpo --load_study True --tag ${experiment_name} --seed $seed2 &
        pid2=$!

    fi
fi

# if resume is True, load the study
if [ "$resume" == 'True' ]; then

    cd ./benchmarking_sim/hpo/${algo}/${experiment_name}
    mysql -u optuna ${algo}_hpo < ${algo}_hpo.sql

    cd ~/safe-control-gym

    # set load_study to True
    python ./examples/hpo/hpo_experiment.py \
                        --algo $algo \
                        --overrides ./benchmarking_sim/${sys}/config_overrides/${sys}_${task}.yaml \
                                    ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                    ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_hpo.yaml \
                        --output_dir ./benchmarking_sim/hpo/${algo} \
                        --sampler $sampler \
                        --use_gpu True \
                        --task ${sys} --func hpo --tag ${experiment_name} --seed $seed3 &
    pid1=$!

    # set load_study to True
    python ./examples/hpo/hpo_experiment.py \
                        --algo $algo \
                        --overrides ./benchmarking_sim/${sys}/config_overrides/${sys}_${task}.yaml \
                                    ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                    ./benchmarking_sim/${sys}/config_overrides/${algo}_${sys}_hpo.yaml \
                        --output_dir ./benchmarking_sim/hpo/${algo} \
                        --sampler $sampler \
                        --use_gpu True \
                        --task ${sys} --func hpo --load_study True --tag ${experiment_name} --seed $seed3 &
    pid2=$!

fi

# move the database from . into output_dir after both commands finish
wait $pid1
echo "job1 finished"
wait $pid2
echo "job2 finished"

# back up first
echo "backing up the database"
mysqldump --no-tablespaces -u optuna ${algo}_hpo > ${algo}_hpo.sql
mv ${algo}_hpo.sql ./benchmarking_sim/hpo/${algo}/${experiment_name}/${algo}_hpo.sql
# remove the database
python ./safe_control_gym/hyperparameters/database.py --func drop --tag ${algo}_hpo