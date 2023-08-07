#!/bin/bash

######## NOTE ########
# 00. Perform HPO (remember to adjust hpo config). 
#     ex: python ./experiments/comparisons/ppo/ppo_experiment.py --func hpo --overrides ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole.yaml ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_hpo.yaml --output_dir ./experiments/comparisons/ppo/hpo --task cartpole --seed 2 --use_gpu True
# if you turn the flag 'perturb_hps' to True, a set of perturbed hps will be produced.
# 1. Specify the folder (PERTURBED_DIR) that contains perturbed hps.
# 2. Change the num_checkpoints as it will be used for evaluation.
# 3. Change the --fun in ppo_experiment.py.
#####################

experiment_name=$1
localOrHost=$2

cd ~/safe-control-gym

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

SEEDS=(22403 84244 98825 40417 58454 47838 56715 77833 19880 59009)

######## Eval Strategy 1 hp sensitivity ########
# PERTURBED_DIR=(./experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s1/seed8_Jul-19-16-29-41_b40566c/hpo/hyperparameters_139.8787/*)
# for HP in "${PERTURBED_DIR[@]}"
# do

# 	echo ${HP}

# 	PERTURBED_HP_FOLDERS=(${HP}/*)
# 	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
# 	do
		
# 		HP_CONFIG=${FOLDER}/*

# 		for SEED in "${SEEDS[@]}" 
# 		do

# 			echo "Training in config:" ${HP_CONFIG} "with seed:" ${SEED} " and outputting dir:" ${FOLDER}

# 			python ./experiments/comparisons/ppo/ppo_experiment.py \
#             --overrides \
#             ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
#             ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
#             --output_dir $FOLDER \
#             --tag run${experiment_name}_s1 \
#             --opt_hps $HP_CONFIG --task cartpole --seed $SEED --use_gpu True

# 		done

# 	done

# done

# for HP in "${PERTURBED_DIR[@]}"
# do

# 	echo ${HP}

# 	PERTURBED_HP_FOLDERS=(${HP}/*)
# 	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
# 	do

# 		algo_seed_dir=(${FOLDER}/run${experiment_name}_s1/seed*)

# 		for algo_seed_dir in "${algo_seed_dir[@]}"
# 		do

# 			echo $algo_seed_dir
# 			python ./experiments/comparisons/ppo/eval.py \
#             --func test_from_checkpoints \
#             --restore ${algo_seed_dir} \
#             --set_test_seed_as_training_eval \
#             --kv_overrides task_config.done_on_out_of_bound=False 
			
# 		done		

# 	done

# done

######## Eval Strategy 2 hp sensitivity ########
PERTURBED_DIR=(./experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s2/seed8_Jul-20-11-45-15_b40566c/hpo/hyperparameters_133.0447/*)
for HP in "${PERTURBED_DIR[@]}"
do

	echo ${HP}

	PERTURBED_HP_FOLDERS=(${HP}/*)
	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
	do
		
		HP_CONFIG=${FOLDER}/*

		for SEED in "${SEEDS[@]}" 
		do

			echo "Training in config:" ${HP_CONFIG} "with seed:" ${SEED} " and outputting dir:" ${FOLDER}

			python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            --output_dir $FOLDER \
            --tag run${experiment_name}_s2 \
            --opt_hps $HP_CONFIG --task cartpole --seed $SEED --use_gpu True

		done

	done

done

for HP in "${PERTURBED_DIR[@]}"
do

	echo ${HP}

	PERTURBED_HP_FOLDERS=(${HP}/*)
	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
	do

		algo_seed_dir=(${FOLDER}/run${experiment_name}_s2/seed*)

		for algo_seed_dir in "${algo_seed_dir[@]}"
		do

			echo $algo_seed_dir
			python ./experiments/comparisons/ppo/eval.py \
            --func test_from_checkpoints \
            --restore ${algo_seed_dir} \
            --set_test_seed_as_training_eval \
            --kv_overrides task_config.done_on_out_of_bound=False 
			
		done		

	done

done

######## Eval Strategy 3 hp sensitivity ########
PERTURBED_DIR=(./experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s3/seed8_Jul-21-05-26-43_b40566c/hpo/hyperparameters_130.2418/*)
for HP in "${PERTURBED_DIR[@]}"
do

	echo ${HP}

	PERTURBED_HP_FOLDERS=(${HP}/*)
	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
	do
		
		HP_CONFIG=${FOLDER}/*

		for SEED in "${SEEDS[@]}" 
		do

			echo "Training in config:" ${HP_CONFIG} "with seed:" ${SEED} " and outputting dir:" ${FOLDER}

			python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            --output_dir $FOLDER \
            --tag run${experiment_name}_s3 \
            --opt_hps $HP_CONFIG --task cartpole --seed $SEED --use_gpu True

		done

	done

done

for HP in "${PERTURBED_DIR[@]}"
do

	echo ${HP}

	PERTURBED_HP_FOLDERS=(${HP}/*)
	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
	do

		algo_seed_dir=(${FOLDER}/run${experiment_name}_s3/seed*)

		for algo_seed_dir in "${algo_seed_dir[@]}"
		do

			echo $algo_seed_dir
			python ./experiments/comparisons/ppo/eval.py \
            --func test_from_checkpoints \
            --restore ${algo_seed_dir} \
            --set_test_seed_as_training_eval \
            --kv_overrides task_config.done_on_out_of_bound=False 
			
		done		

	done

done

######## Eval Strategy 4 hp sensitivity ########
PERTURBED_DIR=(./experiments/comparisons/ppo/hpo/hpo_strategy_study/run1_s4/seed8_Jul-21-22-42-04_b40566c/hpo/hyperparameters_132.9955/*)
for HP in "${PERTURBED_DIR[@]}"
do

	echo ${HP}

	PERTURBED_HP_FOLDERS=(${HP}/*)
	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
	do
		
		HP_CONFIG=${FOLDER}/*

		for SEED in "${SEEDS[@]}" 
		do

			echo "Training in config:" ${HP_CONFIG} "with seed:" ${SEED} " and outputting dir:" ${FOLDER}

			python ./experiments/comparisons/ppo/ppo_experiment.py \
            --overrides \
            ./experiments/comparisons/ppo/config_overrides/cartpole/ppo_cartpole_.yaml \
            ./experiments/comparisons/ppo/config_overrides/cartpole/cartpole_stab.yaml \
            --output_dir $FOLDER \
            --tag run${experiment_name}_s4 \
            --opt_hps $HP_CONFIG --task cartpole --seed $SEED --use_gpu True

		done

	done

done

for HP in "${PERTURBED_DIR[@]}"
do

	echo ${HP}

	PERTURBED_HP_FOLDERS=(${HP}/*)
	for FOLDER in "${PERTURBED_HP_FOLDERS[@]}"
	do

		algo_seed_dir=(${FOLDER}/run${experiment_name}_s4/seed*)

		for algo_seed_dir in "${algo_seed_dir[@]}"
		do

			echo $algo_seed_dir
			python ./experiments/comparisons/ppo/eval.py \
            --func test_from_checkpoints \
            --restore ${algo_seed_dir} \
            --set_test_seed_as_training_eval \
            --kv_overrides task_config.done_on_out_of_bound=False 
			
		done		

	done

done
