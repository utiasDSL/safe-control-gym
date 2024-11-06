import pickle
import numpy as np
from termcolor import colored


data_dir = "examples/mpcacados/temp-data/mpcacados_data_quadrotor_traj_tracking.pkl"

with open(data_dir, "rb") as f:
    data = pickle.load(f)

t_wall = data['trajs_data']['controller_data'][0]['t_wall'][0]
mean_t_wall = np.mean(t_wall)
median_t_wall = np.median(t_wall)
data['metrics']['mean_t_wall_ms'] = mean_t_wall * 1e3  # to milliseconds
data['metrics']['median_t_wall_ms'] = median_t_wall * 1e3

metrics = data['metrics']

for metric_key, metric_val in metrics.items():
    if isinstance(metric_val, list) or isinstance(metric_val, np.ndarray):
        rounded = [f'{elem:.3f}' for elem in metric_val]
        print('{}: {}'.format(colored(metric_key, 'yellow'), rounded))
    else:
        print('{}: {:.3f}'.format(colored(metric_key, 'yellow'), metric_val))