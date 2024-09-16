import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def main():
    ALGO = 'ilqr' # 'ilqr' or 'gpmpc_acados'

    # Load trajectory data
    # state_traj_vizier, mean_state_traj_vizier, action_traj_vizier = aggregate_results(ALGO, 'hp_study_vizier', 'results.pkl')
    state_traj_optuna, mean_state_traj_optuna, action_traj_optuna = aggregate_results(ALGO, 'hp_study_normalized_objective', 'results.pkl')
    # _, _, action_traj_vizier_noise_free = aggregate_results(ALGO, 'hp_study_vizier_nonnoisy', 'results.pkl')
    _, _, action_traj_optuna_noise_free = aggregate_results(ALGO, 'hp_study_normalized_objective_nonnoisy', 'results.pkl')
    # Define improved colors for accessibility (colorblind-friendly palette)
    vizier_color = '#1b9e77'  # Green
    vizier_hull_color = '#a6dbb0'  # Light green for convex hull
    optuna_color = '#7570b3'  # Purple
    optuna_hull_color = '#c7c1e4'  # Light purple for convex hull

    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.plot(action_traj_vizier[:, 0], label='Thrust (Vizier)', color=vizier_color, lw=2, linestyle='-')
    # ax.plot(action_traj_vizier[:, 1], label='Pitch (Vizier)', color=optuna_color, lw=2, linestyle='-')
    ax.plot(action_traj_optuna[:, 0], label='Thrust (Optuna)', color=vizier_hull_color, lw=2, linestyle='--')
    ax.plot(action_traj_optuna[:, 1], label='Pitch (Optuna)', color=optuna_hull_color, lw=2, linestyle='--')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Action')
    ax.legend(loc='best')
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'action.png'))
    # close the plot
    plt.close()

    fix, ax = plt.subplots(figsize=(8, 4))
    # ax.plot(action_traj_vizier_noise_free[:, 0], label='Thrust (Vizier)', color=vizier_color, lw=2, linestyle='-')
    # ax.plot(action_traj_vizier_noise_free[:, 1], label='Pitch (Vizier)', color=optuna_color, lw=2, linestyle='-')
    ax.plot(action_traj_optuna_noise_free[:, 0], label='Thrust (Optuna)', color=vizier_hull_color, lw=2, linestyle='--')
    ax.plot(action_traj_optuna_noise_free[:, 1], label='Pitch (Optuna)', color=optuna_hull_color, lw=2, linestyle='--')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Action')
    ax.legend(loc='best')
    fig.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'action_noise_free.png'))
    plt.close()

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot mean trajectories with labels, adjusting line styles and thickness
    # ax.plot(mean_state_traj_vizier[:, 0], mean_state_traj_vizier[:, 2], label='Vizier', color=vizier_color, lw=2, linestyle='-')
    ax.plot(mean_state_traj_optuna[:, 0], mean_state_traj_optuna[:, 2], label='Optuna', color=optuna_color, lw=2, linestyle='--')
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$z$ [m]')
    ax.set_title('State path in $x$-$z$ plane')


    # Padding factor for convex hull visualization
    k = 1.1

    # # Convex hull visualization with reduced labeling and enhanced alpha blending
    # for i in range(state_traj_vizier.shape[1] - 1):
    #     # Vizier convex hulls
    #     points_of_step_vizier = state_traj_vizier[:, i, [0, 2]]
    #     hull_vizier = ConvexHull(points_of_step_vizier)
    #     cent_vizier = np.mean(points_of_step_vizier, axis=0)
    #     pts_vizier = points_of_step_vizier[hull_vizier.vertices]
    #     poly_vizier = Polygon(k * (pts_vizier - cent_vizier) + cent_vizier, closed=True,
    #                           capstyle='round', facecolor=vizier_hull_color, edgecolor=vizier_color, alpha=0.2)
    #     ax.add_patch(poly_vizier)

    #     points_of_next_step_vizier = state_traj_vizier[:, i + 1, [0, 2]]
    #     points_all_vizier = np.concatenate((points_of_step_vizier, points_of_next_step_vizier), axis=0)
    #     hull_all_vizier = ConvexHull(points_all_vizier)
    #     cent_all_vizier = np.mean(points_all_vizier, axis=0)
    #     pts_all_vizier = points_all_vizier[hull_all_vizier.vertices]
    #     poly_all_vizier = Polygon(k * (pts_all_vizier - cent_all_vizier) + cent_all_vizier, closed=True,
    #                               capstyle='round', facecolor=vizier_hull_color, edgecolor=vizier_color, alpha=0.2)
    #     ax.add_patch(poly_all_vizier)

    for i in range(state_traj_optuna.shape[1] - 1):
        # Optuna convex hulls
        points_of_step_optuna = state_traj_optuna[:, i, [0, 2]]
        hull_optuna = ConvexHull(points_of_step_optuna)
        cent_optuna = np.mean(points_of_step_optuna, axis=0)
        pts_optuna = points_of_step_optuna[hull_optuna.vertices]
        poly_optuna = Polygon(k * (pts_optuna - cent_optuna) + cent_optuna, closed=True,
                              capstyle='round', facecolor=optuna_hull_color, edgecolor=optuna_color, alpha=0.2)
        ax.add_patch(poly_optuna)

        points_of_next_step_optuna = state_traj_optuna[:, i + 1, [0, 2]]
        points_all_optuna = np.concatenate((points_of_step_optuna, points_of_next_step_optuna), axis=0)
        hull_all_optuna = ConvexHull(points_all_optuna)
        cent_all_optuna = np.mean(points_all_optuna, axis=0)
        pts_all_optuna = points_all_optuna[hull_all_optuna.vertices]
        poly_all_optuna = Polygon(k * (pts_all_optuna - cent_all_optuna) + cent_all_optuna, closed=True,
                                  capstyle='round', facecolor=optuna_hull_color, edgecolor=optuna_color, alpha=0.2)
        ax.add_patch(poly_all_optuna)

    # Show only the main trajectories in the legend (no hulls)
    ax.legend(loc='best')
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'state_path.png'))

def aggregate_results(algo, hp_study, pickle_name):
    STUDY_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), algo, hp_study)
    HP_KINDS = os.listdir(STUDY_DIR)
    PICKLE_NAME = pickle_name

    results = {}
    for HP_KIND in HP_KINDS:
        RUNS = os.listdir(os.path.join(STUDY_DIR, HP_KIND))
        results[HP_KIND] = []
        for RUN in RUNS:
            result_path = os.path.join(STUDY_DIR, HP_KIND, RUN, PICKLE_NAME)
            try:
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                    results[HP_KIND].append(result)
            except:
                pass
    
    state_traj = np.array(results['default'][0]['obs']).mean(axis=0, keepdims=True)
    for i in range(1, len(results['default'])):
        state_traj = np.vstack((state_traj, np.array(results['default'][i]['obs']).mean(axis=0, keepdims=True)))

    mean_state_traj = state_traj.mean(axis=0)

    action_traj = np.array(results['default'][0]['current_physical_action'][0])

    return state_traj, mean_state_traj, action_traj


if __name__ == '__main__':
    main()