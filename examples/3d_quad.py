"""3D quadrotor example script.

Example:

    $ python3 3d_quad.py --overrides ./3d_quad.yaml

"""
import os
import time
import yaml
import inspect
import numpy as np
import pybullet as p
import casadi as cs
import matplotlib.pyplot as plt

from safe_control_gym.utils.utils import str2bool
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import PIDController

def main():
    """The main function creating, running, and closing an environment.

    """
    # Set iterations and episode counter.
    num_episodes = 1
    ITERATIONS = int(1000)
    # Start a timer.
    START = time.time()

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    env = make('quadrotor', **config.quadrotor_config)

    # Controller
    ctrl = PIDController()

    # Reset the environment, obtain and print the initial observations.
    initial_obs, initial_info = env.reset()
    # Dynamics info
    print('\nPyBullet dynamics info:')
    print('\t' + str(p.getDynamicsInfo(bodyUniqueId=env.DRONE_IDS[0], linkIndex=-1, physicsClientId=env.PYB_CLIENT)))
    print('\nInitial reset.')
    print('\tInitial observation: ' + str(initial_obs))

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
    t_scaled = np.linspace(t[0], t[-1], env.EPISODE_LEN_SEC*env.CTRL_FREQ)
    ref_x = fx(t_scaled)
    ref_y = fy(t_scaled)
    ref_z = fz(t_scaled)

    # Plot each dimension.
    # plt.plot(t_scaled, x_scaled)
    # plt.plot(t_scaled, x_scaled)
    # plt.plot(t_scaled, x_scaled)
    # plt.show()

    # Plot in 3D.
    ax = plt.axes(projection='3d')
    ax.plot3D(ref_x, ref_y, ref_z)
    ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
    plt.show()

    for i in range(10, ref_x.shape[0], 10):
        p.addUserDebugLine(lineFromXYZ=[ref_x[i-10], ref_y[i-10], ref_z[i-10]],
                           lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                           lineColorRGB=[1, 0, 0],
                           physicsClientId=env.PYB_CLIENT)

    for point in waypoints:
        p.loadURDF(os.path.join(env.URDF_DIR, "gate.urdf"),
                   [point[0], point[1], point[2]-0.05],
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=env.PYB_CLIENT)

    # Run an experiment.
    for i in range(ITERATIONS):
        # Step by keyboard input
        # _ = input('Press any key to continue.')

        # Sample a random action.
        if i == 0:
            action = env.action_space.sample()
        else:
            rpms, _, _ = ctrl.compute_control(control_timestep=env.CTRL_TIMESTEP,
                        cur_pos=np.array([obs[0],obs[2],obs[4]]),
                        cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                        cur_vel=np.array([obs[1],obs[3],obs[5]]),
                        cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                        target_pos=np.array([ref_x[i], ref_y[i], ref_z[i]]),
                        target_vel=np.zeros(3)
                        )
            action = rpms
            action = env.KF * action**2

        # Step the environment and print all returned information.
        obs, reward, done, info = env.step(action)
        #
        print('\n'+str(i)+'-th step.')
        out = '\tApplied action: ' + str(action)
        print(out)
        out = '\tObservation: ' + str(obs)
        print(out)
        out = '\tReward: ' + str(reward)
        print(out)
        out = '\tDone: ' + str(done)
        print(out)
        # out = '\tConstraints evaluations: ' + str(info['constraint_values'])
        # print(out)
        # out = '\tConstraints violation: ' + str(bool(info['constraint_violation']))
        # print(out)

        # If an episode is complete, reset the environment.
        if done:
            num_episodes += 1
            new_initial_obs, new_initial_info = env.reset()
            print(str(num_episodes)+'-th reset.', 7)
            print('Reset obs' + str(new_initial_obs), 2)
            print('Reset info' + str(new_initial_info), 0)

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    out = str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(ITERATIONS, env.CTRL_FREQ, num_episodes, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*env.CTRL_TIMESTEP)/elapsed_sec))
    print(out)


if __name__ == "__main__":
    main()
