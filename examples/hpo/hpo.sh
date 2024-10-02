#!/bin/bash

######## NOTE ########
# This script is used to run HPO in parallel.
# 1. Adjust hpo config.
# 2. Remove or backup the database if needed.
# 3. Create a screen session screen, and detach it Ctrl+a d.
# 4. Run this script by giving experiment name as the first arg, seed as the second, and number of parallel jobs as the third arg.
# 5. If you want to kill them, run pkill -f "python ./.py".
#####################

cd ~/safe-control-gym

experiment_name=$1
seed1=$2
parallel_jobs=$3 # Number of parallel jobs
sampler=$4 # Optuna or Vizier
localOrHost=$5
sys=$6 # cartpole, or quadrotor_2D_attitude
sys_name=${sys%%_*} # cartpole, or quadrotor
algo=$7 # ilqr, gpmpc_acados
prior=$8
safety_filter=$9 # True or False
task=${10} # stab, or tracking
resume=${11} # True or False


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

# echo config path
echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}.yaml"
echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml"
echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml"

# Adjust the seed for each parallel job
seeds=()
for ((i=0; i<parallel_jobs; i++)); do
    seeds[$i]=$((seed1 + i * 100))
done

# if resume is False, create a study for the first job and load it for the remaining jobs
if [ "$resume" == 'False' ]; then
    # First job creates the study
    if [ "$safety_filter" == 'False' ]; then
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
                            --output_dir ./examples/hpo/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys_name} --tag ${experiment_name} --seed ${seeds[0]} &
        pid1=$!

        # wait until the first study is created
        sleep 3

        # Remaining jobs load the study
        for ((i=1; i<parallel_jobs; i++)); do
            python ./examples/hpo/hpo_experiment.py \
                                --algo $algo \
                                --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
                                --output_dir ./examples/hpo/hpo/${algo} \
                                --sampler $sampler \
                                --use_gpu True \
                                --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
            pids[$i]=$!
        done
    fi

    if [ "$safety_filter" == 'True' ]; then
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/linear_mpsc_${sys}_${task}_${prior}.yaml \
                            --kv_overrides sf_config.cost_function=one_step_cost \
                            --output_dir ./examples/hpo/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys_name} --tag ${experiment_name} --seed ${seeds[0]} &
        pid1=$!

        # wait until the first study is created
        sleep 3

        for ((i=1; i<parallel_jobs; i++)); do
            python ./examples/hpo/hpo_experiment.py \
                                --algo $algo \
                                --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
                                            ./examples/hpo/${sys_name}/config_overrides/linear_mpsc_${sys}_${task}_${prior}.yaml \
                                --kv_overrides sf_config.cost_function=one_step_cost \
                                --output_dir ./examples/hpo/hpo/${algo} \
                                --sampler $sampler \
                                --use_gpu True \
                                --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
            pids[$i]=$!
            sleep 3
        done
    fi
fi

# if resume is True, load the study for all jobs
if [ "$resume" == 'True' ]; then
    cd ./examples/hpo/hpo/${algo}/${experiment_name}

    cd ~/safe-control-gym

    for ((i=0; i<parallel_jobs; i++)); do
        python ./examples/hpo/hpo_experiment.py \
                            --algo $algo \
                            --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                                        ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
                            --output_dir ./examples/hpo/hpo/${algo} \
                            --sampler $sampler \
                            --use_gpu True \
                            --task ${sys_name} --load_study True --tag ${experiment_name} --seed ${seeds[$i]} &
        pids[$i]=$!
        sleep 3
    done
fi

# Wait for all jobs to finish
for pid in ${pids[*]}; do
    wait $pid
    echo "Job $pid finished"
done

# back up the database after all jobs finish
echo "backing up the database"
mv ${algo}_hpo.db ./examples/hpo/hpo/${algo}/${experiment_name}/${algo}_hpo_${algo}.db
mv ${algo}_hpo.db-journal ./examples/hpo/hpo/${algo}/${experiment_name}/${algo}_hpo_${algo}.db-journal
mv ${algo}_hpo_endpoint.yaml ./examples/hpo/hpo/${algo}/${experiment_name}/${algo}_hpo_endpoint.yaml
