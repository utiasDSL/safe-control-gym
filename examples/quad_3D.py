'''3D quadrotor example script.

Example:
    $ python3 quad_3D.py --task quadrotor --algo pid --overrides ./quad_3D.yaml
'''

import os
from functools import partial

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from safe_control_gym.experiment import Experiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=True, n_episodes=1, n_steps=None, custom_trajectory=True):
    '''The main function running the 3D quadrotor example.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        custom_trajectory (bool): Whether to run a custom trajectory or a standard one.
    '''

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Set iterations and episode counter.
    ITERATIONS = int(config.task_config['episode_len_sec']*config.task_config['ctrl_freq'])

    # Use function arguments for workflow testing
    config.task_config['gui'] = gui

    env_func = partial(make,
                       config.task,
                       **config.task_config)

    # Setup controller.
    ctrl = make(config.algo,
                    env_func)

    if custom_trajectory:
        # Curve fitting with waypoints.
        waypoints = np.array([(0, 0, 0), (0.2, 0.5, 0.5), (0.5, 0.1, 0.6), (1, 1, 1), (1.3, 1, 1.2)])
        deg = 6
        t = np.arange(waypoints.shape[0])
        fit_x = np.polyfit(t, waypoints[:,0], deg)
        fit_y = np.polyfit(t, waypoints[:,1], deg)
        fit_z = np.polyfit(t, waypoints[:,2], deg)
        fx = np.poly1d(fit_x)
        fy = np.poly1d(fit_y)
        fz = np.poly1d(fit_z)
        t_scaled = np.linspace(t[0], t[-1], ITERATIONS)
        ref_x = fx(t_scaled)
        ref_y = fy(t_scaled)
        ref_z = fz(t_scaled)

        X_GOAL = np.zeros((ITERATIONS, ctrl.env.symbolic.nx))
        X_GOAL[:, 0] = ref_x
        X_GOAL[:, 2] = ref_y
        X_GOAL[:, 4] = ref_z

        ctrl.env.X_GOAL = X_GOAL
        ctrl.reference = X_GOAL

    obs, _ = ctrl.env.reset()

    if config.task_config['gui']:
        # Plot in 3D.
        ax = plt.axes(projection='3d')
        ax.plot3D(ctrl.env.X_GOAL[:, 0], ctrl.env.X_GOAL[:, 2], ctrl.env.X_GOAL[:, 4])
        if custom_trajectory:
            ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
        plt.show()

        for i in range(10, ctrl.env.X_GOAL.shape[0], 10):
            p.addUserDebugLine(lineFromXYZ=[ctrl.env.X_GOAL[i-10,0], ctrl.env.X_GOAL[i-10,2], ctrl.env.X_GOAL[i-10,4]],
                            lineToXYZ=[ctrl.env.X_GOAL[i,0], ctrl.env.X_GOAL[i,2], ctrl.env.X_GOAL[i,4]],
                            lineColorRGB=[1, 0, 0],
                            physicsClientId=ctrl.env.PYB_CLIENT)

        if custom_trajectory:
            for point in waypoints:
                p.loadURDF(os.path.join(ctrl.env.URDF_DIR, 'gate.urdf'),
                        [point[0], point[1], point[2]-0.05],
                        p.getQuaternionFromEuler([0,0,0]),
                        physicsClientId=ctrl.env.PYB_CLIENT)

    # Run the experiment.
    experiment = Experiment(ctrl.env, ctrl)
    trajs_data, _ = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    experiment.close()

    iterations = len(trajs_data['action'][0])
    for i in range(iterations):
        # Step the environment and print all returned information.
        obs, reward, done, info, action = trajs_data['obs'][0][i], trajs_data['reward'][0][i], trajs_data['done'][0][i], trajs_data['info'][0][i], trajs_data['action'][0][i]

        # Print the last action and the information returned at each step.
        print(i, '-th step.')
        print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')

    elapsed_sec = trajs_data['timestamp'][0][-1] - trajs_data['timestamp'][0][0]
    print('\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n'
        .format(iterations, config.task_config.ctrl_freq, 1, elapsed_sec, iterations/elapsed_sec, (iterations*(1. / config.task_config.ctrl_freq))/elapsed_sec))

if __name__ == '__main__':
    run()
