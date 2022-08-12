#!/bin/bash

# Backup the pretrained GP model.
cp trained_gp_model/best_model_0.pth trained_gp_model/bak_best_model_0.pth
cp trained_gp_model/best_model_1.pth trained_gp_model/bak_best_model_1.pth
cp trained_gp_model/best_model_2.pth trained_gp_model/bak_best_model_2.pth
cp trained_gp_model/best_model_3.pth trained_gp_model/bak_best_model_3.pth
cp trained_gp_model/best_model_4.pth trained_gp_model/bak_best_model_4.pth
cp trained_gp_model/best_model_5.pth trained_gp_model/bak_best_model_5.pth
cp trained_gp_model/data.npz trained_gp_model/bak_data.npz

# Re-create the GP models in 'trained_gp_model/' using 800 samples for hyperparameter optimization.
python3 ./gp_mpc_experiment.py --train_only True --task quadrotor --algo gp_mpc --overrides ./config_overrides/gp_mpc_quad_training.yaml
