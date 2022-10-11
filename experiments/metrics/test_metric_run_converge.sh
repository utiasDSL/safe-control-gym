#!/bin/bash

#####################################################
#####################################################
#####################################################
# collect data 

# base data 
python test_metric.py --func collect_data_to_hdf5 --algo random --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/random.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_random.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/pid.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_pid.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/lqr.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_lqr.hdf5 --n_episodes 10 --seed 10

# python test_metric.py --func collect_data_to_hdf5 --algo linear_mpc --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/linear_mpc.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_linear-mpc.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo mpc --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/mpc.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_mpc.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_ppo-100k.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_ppo-1m.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 10 --seed 10


#####################################################
# base data (400 episodes) 
python test_metric.py --func collect_data_to_hdf5 --algo random --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/random.yaml --eval_output_dir temp_data/data --hdf5_file_name base2/quad2d_random.hdf5 --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/pid.yaml --eval_output_dir temp_data/data --hdf5_file_name base2/quad2d_pid.hdf5 --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/lqr.yaml --eval_output_dir temp_data/data --hdf5_file_name base2/quad2d_lqr.hdf5 --n_episodes 100 --seed 10

# python test_metric.py --func collect_data_to_hdf5 --algo linear_mpc --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/linear_mpc.yaml --eval_output_dir temp_data/data --hdf5_file_name base/quad2d_linear-mpc.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo mpc --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/mpc.yaml --eval_output_dir temp_data/data --hdf5_file_name base2/quad2d_mpc.hdf5 --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name base2/quad2d_ppo-100k.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name base2/quad2d_ppo-1m.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 100 --seed 10


#####################################################

























#####################################################
# pid, different prior, vary length data 
python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep10.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep20.hdf5 --n_episodes 20 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep50.hdf5 --n_episodes 50 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep100.hdf5 --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep200.hdf5 --n_episodes 200 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep400.hdf5 --n_episodes 400 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep800.hdf5 --n_episodes 800 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep1200.hdf5 --n_episodes 1200 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo pid --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/pid_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name pid_prior/quad2d_pid-prior1.5x_ep1600.hdf5 --n_episodes 1600 --seed 10


# compute metric 
python test_metric.py --func test_metric_convergence --overrides config_overrides/quad_track_2d_data/vary_metric/pid_prior1.5x.yaml --metric geom_loss --geom_loss_func sinkhorn --geom_loss_blur 0.01 --not_include_action --eval_output_dir temp_data/data --file_name pid_prior.yaml --plot --fig_name pid_prior.png --metric_name Wasserstein



#####################################################
# lqr, different prior, vary length data 
python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep10.hdf5 --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep20.hdf5 --n_episodes 20 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep50.hdf5 --n_episodes 50 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep100.hdf5 --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep200.hdf5 --n_episodes 200 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep400.hdf5 --n_episodes 400 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep800.hdf5 --n_episodes 800 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep1200.hdf5 --n_episodes 1200 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo lqr --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/lqr_prior1.5x.yaml --eval_output_dir temp_data/data --hdf5_file_name lqr_prior/quad2d_lqr-prior1.5x_ep1600.hdf5 --n_episodes 1600 --seed 10


# compute metric 
python test_metric.py --func test_metric_convergence --overrides config_overrides/quad_track_2d_data/vary_metric/lqr_prior1.5x.yaml --metric geom_loss --geom_loss_func sinkhorn --geom_loss_blur 0.01 --not_include_action --eval_output_dir temp_data/data --file_name lqr_prior.yaml --plot --fig_name lqr_prior.png --metric_name Wasserstein





#####################################################
# ppo, different prior, vary length data 
python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep10.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep20.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 20 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep50.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 50 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep100.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep200.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 200 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep400.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 400 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo100k_prior/quad2d_ep800.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint checkpoints/model_100000.pt --n_episodes 800 --seed 10


# compute metric 
python test_metric.py --func test_metric_convergence --overrides config_overrides/quad_track_2d_data/vary_metric/ppo100k.yaml --metric geom_loss --geom_loss_func sinkhorn --geom_loss_blur 0.01 --not_include_action --eval_output_dir temp_data/data --file_name ppo100k_prior.yaml --plot --fig_name ppo100k_prior.png --metric_name Wasserstein


#####################################################
# ppo (trained), different prior, vary length data 
python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep10.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 10 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep20.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 20 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep50.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 50 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep100.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 100 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep200.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 200 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep400.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 400 --seed 10

python test_metric.py --func collect_data_to_hdf5 --algo ppo --task quadrotor --overrides config_overrides/quad_track_2d_data/base.yaml config_overrides/quad_track_2d_data/vary/ppo.yaml --eval_output_dir temp_data/data --hdf5_file_name ppo5m_prior/quad2d_ep800.hdf5 --restore temp_data/quad_track_2d_box/ppo/seed668_Sep-08-16-09-45_v0.5.0-345-g6a5a455/ --checkpoint model_latest.pt --n_episodes 800 --seed 10


# compute metric 
python test_metric.py --func test_metric_convergence --overrides config_overrides/quad_track_2d_data/vary_metric/ppo5m.yaml --metric geom_loss --geom_loss_func sinkhorn --geom_loss_blur 0.01 --not_include_action --eval_output_dir temp_data/data --file_name ppo5m_prior.yaml --plot --fig_name ppo5m_prior.png --metric_name Wasserstein