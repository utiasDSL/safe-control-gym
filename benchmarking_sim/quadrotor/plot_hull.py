
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from safe_control_gym.utils.configuration import ConfigFactory
from functools import partial
from safe_control_gym.utils.registration import make

# # get the config
# ALGO = 'mpc_acados'
# SYS = 'quadrotor_2D_attitude'
# TASK = 'tracking'
# # PRIOR = '200'
# PRIOR = '100'
# agent = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS
# SAFETY_FILTER = None

# # check if the config file exists
# assert os.path.exists(f'./config_overrides/{SYS}_{TASK}.yaml'), f'./config_overrides/{SYS}_{TASK}.yaml does not exist'
# assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
# if SAFETY_FILTER is None:
#     sys.argv[1:] = ['--algo', ALGO,
#                     '--task', agent,
#                     '--overrides',
#                         f'./config_overrides/{SYS}_{TASK}.yaml',
#                         f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
#                     '--seed', '2',
#                     '--use_gpu', 'True',
#                     '--output_dir', f'./{ALGO}/results',
#                         ]
# fac = ConfigFactory()
# fac.add_argument('--func', type=str, default='train', help='main function to run.')
# fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
# # merge config and create output directory
# config = fac.merge()
# # Create an environment
# env_func = partial(make,
#                     config.task,
#                     seed=config.seed,
#                     **config.task_config
#                     )
# random_env = env_func(gui=False)
# X_GOAL = random_env.X_GOAL
# print('X_GOAL.shape', X_GOAL.shape)

# get the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

script_path = os.path.dirname(os.path.realpath(__file__))
#############################################
# generalization = False
generalization = True
#############################################
# if not generalization:
#     gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_rti/temp'
# if generalization:
#     gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/100_300_rti_rollout/temp'
# # gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_rti/temp'
# # get all directories in the gp_model_path
# gp_model_dirs = [d for d in os.listdir(gp_model_path) if os.path.isdir(os.path.join(gp_model_path, d))]
# gp_model_dirs = [os.path.join(gp_model_path, d) for d in gp_model_dirs]

# traj_data_name = 'gpmpc_acados_data_quadrotor_traj_tracking.pkl'
# data_name = [os.path.join(d, traj_data_name) for d in gp_model_dirs]

# # print(data_name)
# # data = np.load(data_name[0], allow_pickle=True)
# # print(data.keys())
# # print(data['trajs_data'].keys())
# # print(data['trajs_data']['obs'][0].shape) # (541, 6)
# data = []
# for d in data_name:
#     data.append(np.load(d, allow_pickle=True))
# gpmpc_traj_data = [d['trajs_data']['obs'][0] for d in data]
# gpmpc_traj_data = np.array(gpmpc_traj_data)
# print(gpmpc_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# # take average of all seeds
# mean_traj_data = np.mean(gpmpc_traj_data, axis=0)
# print(mean_traj_data.shape) # (mean_541, 6)

# ### plot the ilqr data
# if not generalization:
#     ilqr_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/ilqr/results/temp'
# if generalization:
#     ilqr_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/ilqr/results_rollout/temp'

# ilqr_data_dirs = [d for d in os.listdir(ilqr_data_path) if os.path.isdir(os.path.join(ilqr_data_path, d))]
# ilqr_traj_data_name = 'ilqr_data_quadrotor_traj_tracking.pkl'
# ilqr_traj_data_name = [os.path.join(d, ilqr_traj_data_name) for d in ilqr_data_dirs]

# ilqr_data = []
# for d in ilqr_traj_data_name:
#     ilqr_data.append(np.load(os.path.join(ilqr_data_path, d), allow_pickle=True))
# ilqr_traj_data = [d['trajs_data']['obs'][0] for d in ilqr_data]
# ilqr_traj_data = np.array(ilqr_traj_data)
# print(ilqr_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# # take average of all seeds
# ilqr_mean_traj_data = np.mean(ilqr_traj_data, axis=0)
# print(ilqr_mean_traj_data.shape) # (mean_541, 6)

# load ppo and sac data
if not generalization:
    ppo_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/PPO_traj.npy'
else:
    ppo_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/PPO_traj_gen.npy'
ppo_data = np.load(ppo_data_path, allow_pickle=True).item()
print(ppo_data.keys()) # (x, 541, 6) seed, time_step, obs
print(ppo_data['obs'][0].shape)
ppo_traj_data = np.array(ppo_data['obs'])
print(ppo_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
ppo_mean_traj_data = np.mean(ppo_traj_data, axis=0)[:, 0:6]
print(ppo_mean_traj_data.shape) # (mean_541, 6)

if not generalization:
    sac_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/SAC_traj.npy'
else:
    sac_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/data/SAC_traj_gen.npy'
sac_data = np.load(sac_data_path, allow_pickle=True).item()
print(sac_data.keys()) # (x, 541, 6) seed, time_step, obs
print(sac_data['obs'][0].shape)
sac_traj_data = np.array(sac_data['obs'])
print(sac_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
sac_mean_traj_data = np.mean(sac_traj_data, axis=0)[:, 0:6]
print(sac_mean_traj_data.shape) # (mean_541, 6)

##################################################
# plotting trajectory
gpmpc_color = 'blue'
# gpmpc_hull_color = 'lightskyblue'
gpmpc_hull_color = 'cornflowerblue'
ilqr_color = 'gray'
ilqr_hull_color = 'lightgray'
ppo_color = 'orange'
ppo_hull_color = 'peachpuff'
sac_color = 'green'
sac_hull_color = 'lightgreen'
ref_color = 'black'
##################################################
# plot the state path x, z [0, 2]
# mean_points = mean_traj_data[:, [0, 2]]
# mean_points_ilqr = ilqr_mean_traj_data[:, [0, 2]]
fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(X_GOAL[:, 0], X_GOAL[:, 2], color=ref_color, linestyle='-.', label='Reference')
# ax.plot(mean_points_ilqr[:,0], mean_points_ilqr[:,1], label='iLQR', color=ilqr_color)
# ax.plot(mean_points[:,0], mean_points[:,1], label='GP-MPC', color=gpmpc_color)
ax.plot(ppo_mean_traj_data[:,0], ppo_mean_traj_data[:,2], label='PPO', color=ppo_color)
ax.plot(sac_mean_traj_data[:,0], sac_mean_traj_data[:,2], label='SAC', color=sac_color)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$z$ [m]')
ax.set_title('State path in $x$-$z$ plane')
# set the super title
if not generalization:
    fig.suptitle('Figure-eight reference tracking', fontsize=18)
else:
    fig.suptitle('Generalization to an unseen task', fontsize=18)
fig.tight_layout()
ax.legend(ncol=5, loc='upper center')


# plot the convex hull of each steps
k = 1.1 # padding factor
for i in range(ppo_traj_data.shape[1] - 1):
# # for i in range(1):
#     # ilqr
#     points_of_step_ilqr = ilqr_traj_data[:, i, [0, 2]]
#     hull_ilqr = ConvexHull(points_of_step_ilqr)
#     cent_ilqr = np.mean(points_of_step_ilqr, axis=0)
#     pts_ilqr = points_of_step_ilqr[hull_ilqr.vertices]

#     poly_ilqr = Polygon(k*(pts_ilqr - cent_ilqr) + cent_ilqr, closed=True,
#                 capstyle='round', facecolor=ilqr_hull_color, alpha=1.0)
#     ax.add_patch(poly_ilqr)

#     points_of_next_step_ilqr = ilqr_traj_data[:, i+1, [0, 2]]
#     points_all_ilqr = np.concatenate((points_of_step_ilqr, points_of_next_step_ilqr), axis=0)
#     hull_all_ilqr = ConvexHull(points_all_ilqr)
#     cent_all_ilqr = np.mean(points_all_ilqr, axis=0)
#     pts_all_ilqr = points_all_ilqr[hull_all_ilqr.vertices]
#     poly_all_ilqr = Polygon(k*(pts_all_ilqr - cent_all_ilqr) + cent_all_ilqr, closed=True,
#                 capstyle='round', facecolor=ilqr_hull_color, alpha=1.0)
#     ax.add_patch(poly_all_ilqr)
    
    '''
    NOTE: The current color choice is not ideal in the sense that 
    overlapping the same color will make the color darker.
    Therefore, alpha of each convex hull is set to 1.0. This will 
    resutls in different convex hulls overlapping each other and 
    the one in the bottom will not be visible.
    '''

    # sac
    # plot the convex hull of each steps
    points_of_step_sac = sac_traj_data[:, i, [0, 2]]
    hull_sac = ConvexHull(points_of_step_sac)
    cent_sac = np.mean(points_of_step_sac, axis=0)
    pts_sac = points_of_step_sac[hull_sac.vertices]
    poly_sac = Polygon(k*(pts_sac - cent_sac) + cent_sac, closed=True,
                capstyle='round', facecolor=sac_hull_color, alpha=1.0)
    ax.add_patch(poly_sac)
    
    # also connect the points of the next step
    points_of_next_step_sac = sac_traj_data[:, i+1, [0, 2]]
    points_all_sac = np.concatenate((points_of_step_sac, points_of_next_step_sac), axis=0)
    hull_all_sac = ConvexHull(points_all_sac)
    cent_all_sac = np.mean(points_all_sac, axis=0)
    pts_all_sac = points_all_sac[hull_all_sac.vertices]
    poly_all_sac = Polygon(k*(pts_all_sac - cent_all_sac) + cent_all_sac, closed=True,
                capstyle='round', facecolor=sac_hull_color, alpha=1.0)
    ax.add_patch(poly_all_sac)

    # ppo
    points_of_step_ppo = ppo_traj_data[:, i, [0, 2]]
    hull_ppo = ConvexHull(points_of_step_ppo)
    cent_ppo = np.mean(points_of_step_ppo, axis=0)
    pts_ppo = points_of_step_ppo[hull_ppo.vertices]
    poly_ppo = Polygon(k*(pts_ppo - cent_ppo) + cent_ppo, closed=True,
                   capstyle='round', facecolor=ppo_hull_color, alpha=1.0)
    ax.add_patch(poly_ppo)

    points_of_next_step_ppo = ppo_traj_data[:, i+1, [0, 2]]
    points_all_ppo = np.concatenate((points_of_step_ppo, points_of_next_step_ppo), axis=0)
    hull_all_ppo = ConvexHull(points_all_ppo)
    cent_all_ppo = np.mean(points_all_ppo, axis=0)
    pts_all_ppo = points_all_ppo[hull_all_ppo.vertices]
    poly_all_ppo = Polygon(k*(pts_all_ppo - cent_all_ppo) + cent_all_ppo, closed=True,
                   capstyle='round', facecolor=ppo_hull_color, alpha=1.0)

    # # gpmpc
    # points_of_step = gpmpc_traj_data[:, i, [0, 2]]
    # hull = ConvexHull(points_of_step)
    # cent = np.mean(points_of_step, axis=0)
    # pts = points_of_step[hull.vertices]
    # poly = Polygon(k*(pts - cent) + cent, closed=True,
    #                capstyle='round', facecolor=gpmpc_hull_color, alpha=1.0)
    # ax.add_patch(poly)

    # # also connect the points of the next step
    # points_of_next_step = gpmpc_traj_data[:, i+1, [0, 2]]
    # points_all = np.concatenate((points_of_step, points_of_next_step), axis=0)
    # hull_all = ConvexHull(points_all)
    # cent_all = np.mean(points_all, axis=0)
    # pts_all = points_all[hull_all.vertices]
    # poly_all = Polygon(k*(pts_all - cent_all) + cent_all, closed=True,
    #                capstyle='round', facecolor=gpmpc_hull_color, alpha=1.0)
    # ax.add_patch(poly_all)



if not generalization:
    fig.savefig(os.path.join(script_path, 'xz_path_performance.png'))
    print(f'Saved at {os.path.join(script_path, "xz_path_performance.png")}')
else:
    fig.savefig(os.path.join(script_path, 'xz_path_generalization.png'))
    print(f'Saved at {os.path.join(script_path, "xz_path_generalization.png")}')


