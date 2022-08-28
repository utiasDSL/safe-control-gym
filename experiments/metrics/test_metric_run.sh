#!/bin/bash

#####################################################

# LSED
python test_metric.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric lsed --eval_output_dir temp_data/quad_track_2d/lsed --seed 78 --plot

# DTW
python test_metric.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric dtw --eval_output_dir temp_data/quad_track_2d/dtw --seed 78 --plot

# EDR
python test_metric.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric edr --eval_output_dir temp_data/quad_track_2d/edr --seed 78 --plot

# LCSS
python test_metric.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric lcss --eval_output_dir temp_data/quad_track_2d/lcss --seed 78 --plot

# Frechet
python test_metric.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric discrete_frechet --eval_output_dir temp_data/quad_track_2d/discrete_frechet --seed 78 --plot

# MMD 
python test_metric.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric mmd_loss --eval_output_dir temp_data/quad_track_2d/mmd --seed 78 --plot








#####################################################
#####################################################
#####################################################
# batch runs

#####################################################
# quad_track_2d, pid

# LSED
bash run_metric.sh test_metric quad_track_2d pid quadrotor lsed - 
#
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/lsed --file_name temp.yaml --metric lsed 

# DTW
bash run_metric.sh test_metric quad_track_2d pid quadrotor dtw - 
#
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/dtw --file_name temp.yaml --metric dtw 

# EDR
bash run_metric.sh test_metric quad_track_2d pid quadrotor edr - 
#
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/edr --file_name temp.yaml --metric edr 

# LCSS
bash run_metric.sh test_metric quad_track_2d pid quadrotor lcss - 
#
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/lcss --file_name temp.yaml --metric lcss 

# Frechet
bash run_metric.sh test_metric quad_track_2d pid quadrotor discrete_frechet - 
#
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/discrete_frechet --file_name temp.yaml --metric discrete_frechet 

# MMD (gaussian)
bash run_metric.sh test_metric quad_track_2d pid quadrotor mmd_loss - 
# nonparallel to save space
bash run_metric.sh test_metric_nonparallel quad_track_2d pid quadrotor mmd_loss -
#
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/mmd_loss --file_name temp.yaml --metric mmd_loss 

# Wasserstein 
# nonparallel to save space
bash run_metric.sh test_metric_nonparallel quad_track_2d pid quadrotor geom_loss wasserstein "--geom_loss_func sinkhorn --geom_loss_blur 0.01"
# 
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/wasserstein --file_name temp.yaml --metric wasserstein

# Hausdorff/ICP (iterative closest point)
# nonparallel to save space
bash run_metric.sh test_metric_nonparallel quad_track_2d pid quadrotor geom_loss hausdorff "--geom_loss_func hausdorff --geom_loss_blur 0.01 --geom_loss_kernel gaussian"
# 
python test_metric.py --func plot_metric --eval_output_dir temp_data/quad_track_2d/hausdorff --file_name temp.yaml --metric hausdorff



## plot correlations as a table
python test_metric.py --func plot_correlation --eval_output_dir temp_data/quad_track_2d --file_name temp.yaml --csv_file_name temp.csv
