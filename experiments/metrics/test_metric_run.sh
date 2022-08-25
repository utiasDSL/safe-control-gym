

#####################################################

# DTW
python test_metric2.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric dtw --eval_output_dir temp_data/quad_track_2d/dtw --seed 78 --plot

# EDR
python test_metric2.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric edr --eval_output_dir temp_data/quad_track_2d/edr --seed 78 --plot

# LCSS
python test_metric2.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric lcss --eval_output_dir temp_data/quad_track_2d/lcss --seed 78 --plot

# Frechet
python test_metric2.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric discrete_frechet --eval_output_dir temp_data/quad_track_2d/discrete_frechet --seed 78 --plot

# MMD 
python test_metric2.py --func test_trajectory_metric --algo pid --task quadrotor --overrides config_overrides/quad_track_2d/base.yaml config_overrides/quad_track_2d/variant.yaml --n_episodes 10 --metric mmd_loss --eval_output_dir temp_data/quad_track_2d/mmd --seed 78 --plot
