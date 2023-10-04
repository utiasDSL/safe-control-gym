#!/bin/bash

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stab'
# TASK='track'

ALGO='ppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/

if [ "$ALGO" == 'safe_explorer_ppo' ]; then
    # Pretrain the unsafe controller/agent.
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}_pretrain.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./unsafe_rl_temp_data/ \
        --seed 2 \
        --kv_overrides \
            task_config.init_state=None

    # Move the newly trained unsafe model.
    mv ./unsafe_rl_temp_data/model_latest.pt ./models/${ALGO}/${ALGO}_pretrain_${SYS}_${TASK}.pt

    # Removed the temporary data used to train the new unsafe model.
    rm -r -f ./unsafe_rl_temp_data/
fi

# Train the unsafe controller/agent.
python3 ../../safe_control_gym/experiments/train_rl_controller.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --overrides \
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
    --output_dir ./unsafe_rl_temp_data/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        task_config.randomized_init=True \
        algo_config.pretrained=./models/${ALGO}/${ALGO}_pretrain_${SYS}_${TASK}.pt

# Move the newly trained unsafe model.
mv ./unsafe_rl_temp_data/model_best.pt ./models/${ALGO}/${ALGO}_model_${SYS}_${TASK}.pt

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/
