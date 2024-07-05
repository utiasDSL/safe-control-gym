#!/bin/bash

SYS='cartpole'
#SYS='quadrotor_2D'
#SYS='quadrotor_2D_attitude'
#SYS='quadrotor_3D'

TASK='stab'
#TASK='track'

#ALGO='ppo'
ALGO='sac'
#ALGO='td3'
#ALGO='ddpg'

#ALGO='safe_explorer_ppo'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Removed the temporary data used to train the new unsafe model.
# rm -r -f ./${ALGO}_data_2/



# Train the unsafe controller/agent.
python ./hpo_experiment.py \
                    --algo ${ALGO} \
                    --overrides \
                    ./rl/${ALGO}/config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                    ./rl/config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                    --output_dir ./Results/${SYS}_${ALGO}_data/ \
                    --tag Test \
                    --opt_hps ./rl/${ALGO}/config_overrides/${SYS}/optimized_hyperparameters.yaml \
                    --task ${SYS} --seed 2 \
                    --use_gpu True

# Removed the temporary data used to train the new unsafe model.
#rm -r -f ./unsafe_rl_temp_data/
