"""Base environment class module.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.

"""
import os
from enum import Enum
from matplotlib import pyplot as plt
import numpy as np
import gym
from gym.utils import seeding
import copy


class Cost(str, Enum):
    """Reward/cost functions enumeration class.

    """

    RL_REWARD = "rl_reward"  # Default RL reward function.
    QUADRATIC = "quadratic"  # Quadratic cost.


class Task(str, Enum):
    """Environment tasks enumeration class.

    """

    STABILIZATION = "stabilization"  # Stabilization task.
    TRAJ_TRACKING = "traj_tracking"  # Trajectory tracking task.


class BenchmarkEnv(gym.Env):
    """Benchmark environment base class.
    
    Attributes:
        id (int): unique identifier of the current env instance (among other instances). 

    """
    _count = 0  # Class variable, count env instance in current process.

    def __init__(self,
                 output_dir=None,
                 seed: int = 0,
                 info_in_reset: bool = False,
                 episode_len_sec: int = 5,
                 cost: Cost = Cost.RL_REWARD
                 ):
        """Initialization method for BenchmarkEnv.

        Args:
            seed (int, optional): Seed for the random number generator.
            info_in_reset (bool, optional): Whether .reset() returns a dictionary with the
                                            environment's symbolic model.
            episode_len_sec (int, optional): Maximum episode duration in seconds.
            cost (Cost, optional): Cost function choice used to compute the reward in .step().

        """
        # Assign unique ID based on env instance count.
        self.id = self.__class__._count
        self.__class__._count += 1
        # Directory to save any env output.
        if output_dir is None:
            output_dir = os.getcwd()
        self.output_dir = output_dir
        # Default seed None means pure randomness/no seeding.
        self.seed(seed)
        self.INFO_IN_RESET = info_in_reset
        self.initial_reset = False
        # Maximum episode length in seconds.
        self.EPISODE_LEN_SEC = episode_len_sec
        # Define cost-related quantities.
        self.COST = Cost(cost)
        # Default Q and R matrices for quadratic cost.
        if self.COST == Cost.QUADRATIC:
            self.Q = np.eye(self.observation_space.shape[0])
            self.R = np.eye(self.action_space.shape[0])
        # Custom setups.
        self._setup_symbolic()
        self._setup_disturbances()
        self._setup_constraints()

    def _check_initial_reset(self):
        """Makes sure that .reset() is called at least once before .step()."""
        if not self.initial_reset:
            raise RuntimeError(
                "[ERROR] You must call env.reset() at least once before using env.step()."
            )

    def _randomize_values_by_info(self,
                                  original_values,
                                  randomization_info
                                  ):
        """Randomizes a list of values according to desired distributions.

        Args:
            original_values (dict): a dict of orginal values.
            randomization_info (dict): A dictionary containing information about the distributions
                                       used to randomize original_values.

        Returns:
            dict: A dict of randomized values.

        """
        # Start from a copy of the original values.
        randomized_values = copy.deepcopy(original_values)
        # Copy the info dict to parse it with "pop".
        rand_info_copy = copy.deepcopy(randomization_info)
        # Randomized and replace values for which randomization info are given.
        for key in original_values:
            if key in rand_info_copy:
                # Get distribution removing it from info dict.
                distrib = getattr(self.np_random,
                                  rand_info_copy[key].pop("distrib"))
                # Pop positional args.
                d_args = rand_info_copy[key].pop("args", [])
                # Keyword args are just anything left.
                d_kwargs = rand_info_copy[key]
                # Randomize.
                randomized_values[key] = distrib(*d_args, **d_kwargs)
        return randomized_values

    def seed(self,
             seed=None
             ):
        """Sets up a random number generator for a given seed.
        
        Current convention: non-positive seed same as None 

        """
        if seed is not None:
            seed = seed if seed > 0 else None
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _setup_symbolic(self):
        """Creates a symbolic (CasADi) model for dynamics and cost."""
        raise NotImplementedError

    def _setup_disturbances(self):
        """Creates attributes and action spaces for the disturbances."""
        raise NotImplementedError

    def _setup_constraints(self):
        """Creates a list of constraints as an attribute."""
        raise NotImplementedError

    def _generate_trajectory(self,
                             traj_type="figure8",
                             traj_length=10.0,
                             num_cycles=1,
                             traj_plane="xy",
                             position_offset=np.array([0, 0]),
                             scaling=1.0,
                             sample_time=0.01):
        """Generates a 2D trajectory.

        Args:
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_length (float, optional): The length of the trajectory in seconds.
            num_cycles (int, optional): The number of cycles within the length.
            traj_plane (str, optional): The plane of the trajectory (e.g. "xz").
            position_offset (ndarray, optional): An initial position offset in the plane.
            scaling (float, optional): Scaling factor for the trajectory.
            sample_time (float, optional): The sampling timestep of the trajectory.

        Returns:
            ndarray: The positions in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The velocities in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The scalar speed of the trajectory sampled for its entire duration.

        """
        # Get trajectory type.
        valid_traj_type = ["circle", "square", "figure8"]
        if traj_type not in valid_traj_type:
            raise ValueError("Trajectory type should be one of [circle, square, figure8].")
        traj_period = traj_length / num_cycles
        direction_list = ["x", "y", "z"]
        # Get coordinates indexes.
        if traj_plane[0] in direction_list and traj_plane[
                1] in direction_list and traj_plane[0] != traj_plane[1]:
            coord_index_a = direction_list.index(traj_plane[0])
            coord_index_b = direction_list.index(traj_plane[1])
        else:
            raise ValueError("Trajectory plane should be in form of ab, where a and b can be {x, y, z}.")
        # Generate time stamps.
        times = np.arange(0, traj_length, sample_time)
        pos_ref_traj = np.zeros((len(times), 3))
        vel_ref_traj = np.zeros((len(times), 3))
        speed_traj = np.zeros((len(times), 1))
        # Compute trajectory points.
        for t in enumerate(times):
            pos_ref_traj[t[0]], vel_ref_traj[t[0]] = self._get_coordinates(t[1],
                                                                           traj_type,
                                                                           traj_period,
                                                                           coord_index_a,
                                                                           coord_index_b,
                                                                           position_offset[0],
                                                                           position_offset[1],
                                                                           scaling)
            speed_traj[t[0]] = np.linalg.norm(vel_ref_traj[t[0]])
        return pos_ref_traj, vel_ref_traj, speed_traj

    def _get_coordinates(self,
                         t,
                         traj_type,
                         traj_period,
                         coord_index_a,
                         coord_index_b,
                         position_offset_a,
                         position_offset_b,
                         scaling
                         ):
        """Computes the coordinates of a specified trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_period (float): The period of the trajectory in seconds.
            coord_index_a (int): The index of the first coordinate of the trajectory plane.
            coord_index_b (int): The index of the second coordinate of the trajectory plane.
            position_offset_a (float): The offset in the first coordinate of the trajectory plane.
            position_offset_b (float): The offset in the second coordinate of the trajectory plane.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            ndarray: The position in x, y, z, at time t.
            ndarray: The velocity in x, y, z, at time t.

        """
        # Get coordinates for the trajectory chosen.
        if traj_type == "figure8":
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._figure8(
                t, traj_period, scaling)
        elif traj_type == "circle":
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._circle(
                t, traj_period, scaling)
        elif traj_type == "square":
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._square(
                t, traj_period, scaling)
        # Initialize position and velocity references.
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        # Set position and velocity references based on the plane of the trajectory chosen.
        pos_ref[coord_index_a] = coords_a + position_offset_a
        vel_ref[coord_index_a] = coords_a_dot
        pos_ref[coord_index_b] = coords_b + position_offset_b
        vel_ref[coord_index_b] = coords_b_dot
        return pos_ref, vel_ref

    def _figure8(self,
                 t,
                 traj_period,
                 scaling
                 ):
        """Computes the coordinates of a figure8 trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            float: The position in the first coordinate. 
            float: The position in the second coordinate. 
            float: The velocity in the first coordinate. 
            float: The velocity in the second coordinate. 

        """
        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.sin(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t) * np.cos(traj_freq * t)
        coords_a_dot = scaling * traj_freq * np.cos(traj_freq * t)
        coords_b_dot = scaling * traj_freq * (np.cos(traj_freq * t)**2 - np.sin(traj_freq * t)**2)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _circle(self,
                t,
                traj_period,
                scaling
                ):
        """Computes the coordinates of a circle trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            float: The position in the first coordinate. 
            float: The position in the second coordinate. 
            float: The velocity in the first coordinate. 
            float: The velocity in the second coordinate. 

        """
        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.cos(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t)
        coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
        coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _square(self,
                t,
                traj_period,
                scaling
                ):
        """Computes the coordinates of a square trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            float: The position in the first coordinate. 
            float: The position in the second coordinate. 
            float: The velocity in the first coordinate. 
            float: The velocity in the second coordinate. 

        """
        # Compute time for each segment to complete.
        segment_period = traj_period / 4.0
        traverse_speed = scaling / segment_period
        # Compute time for the cycle.
        cycle_time = t % traj_period
        # Check time along the current segment and ratio of completion.
        segment_time = cycle_time % segment_period
        # Check current segment index.
        segment_index = int(np.floor(cycle_time / segment_period))
        # Position along segment
        segment_position = traverse_speed * segment_time
        if segment_index == 0:
            # Moving up along second axis from (0, 0).
            coords_a = 0.0
            coords_b = segment_position
            coords_a_dot = 0.0
            coords_b_dot = traverse_speed
        elif segment_index == 1:
            # Moving left along first axis from (0, 1).
            coords_a = -segment_position
            coords_b = scaling
            coords_a_dot = -traverse_speed
            coords_b_dot = 0.0
        elif segment_index == 2:
            # Moving down along second axis from (-1, 1).
            coords_a = -scaling
            coords_b = scaling - segment_position
            coords_a_dot = 0.0
            coords_b_dot = -traverse_speed
        elif segment_index == 3:
            # Moving right along second axis from (-1, 0).
            coords_a = -scaling + segment_position
            coords_b = 0.0
            coords_a_dot = traverse_speed
            coords_b_dot = 0.0
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _plot_trajectory(self,
                         traj_type,
                         traj_plane,
                         traj_length,
                         num_cycles,
                         pos_ref_traj,
                         vel_ref_traj,
                         speed_traj
                         ):
        """Plots a trajectory along x, y, z, and in a 3D projection.

        Args:
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_plane (str, optional): The plane of the trajectory (e.g. "xz").
            traj_length (float, optional): The length of the trajectory in seconds.
            num_cycles (int, optional): The number of cycles within the length.
            pos_ref_traj (ndarray): The positions in x, y, z of the trajectory sampled for its entire duration.
            vel_ref_traj (ndarray): The velocities in x, y, z of the trajectory sampled for its entire duration.
            speed_traj (ndarray): The scalar speed of the trajectory sampled for its entire duration.

        """
        # Print basic properties.
        print("Trajectory type: %s" % traj_type)
        print("Trajectory plane: %s" % traj_plane)
        print("Trajectory length: %s sec" % traj_length)
        print("Number of cycles: %d" % num_cycles)
        print("Trajectory period: %.2f sec" % (traj_length / num_cycles))
        print("Angular speed: %.2f rad/sec" % (2.0 * np.pi / (traj_length / num_cycles)))
        print(
            "Position bounds: x [%.2f, %.2f] m, y [%.2f, %.2f] m, z [%.2f, %.2f] m"
            % (min(pos_ref_traj[:, 0]), max(pos_ref_traj[:, 0]),
               min(pos_ref_traj[:, 1]), max(pos_ref_traj[:, 1]),
               min(pos_ref_traj[:, 2]), max(pos_ref_traj[:, 2])))
        print(
            "Velocity bounds: vx [%.2f, %.2f] m/s, vy [%.2f, %.2f] m/s, vz [%.2f, %.2f] m/s"
            % (min(vel_ref_traj[:, 0]), max(vel_ref_traj[:, 0]),
               min(vel_ref_traj[:, 1]), max(vel_ref_traj[:, 1]),
               min(vel_ref_traj[:, 2]), max(vel_ref_traj[:, 2])))
        print("Speed: min %.2f m/s max %.2f m/s mean %.2f" %
              (min(speed_traj), max(speed_traj), np.mean(speed_traj)))
        # Plot in x, y, z.
        fig, axs = plt.subplots(3, 2)
        t = np.arange(0, traj_length, traj_length / pos_ref_traj.shape[0])
        axs[0, 0].plot(t, pos_ref_traj[:, 0])
        axs[0, 0].set_ylabel('pos x (m)')
        axs[1, 0].plot(t, pos_ref_traj[:, 1])
        axs[1, 0].set_ylabel('pos y (m)')
        axs[2, 0].plot(t, pos_ref_traj[:, 2])
        axs[2, 0].set_ylabel('pos z (m)')
        axs[2, 0].set_xlabel('time (s)')
        axs[0, 1].plot(t, vel_ref_traj[:, 0])
        axs[0, 1].set_ylabel('vel x (m)')
        axs[1, 1].plot(t, vel_ref_traj[:, 1])
        axs[1, 1].set_ylabel('vel y (m)')
        axs[2, 1].plot(t, vel_ref_traj[:, 2])
        axs[2, 1].set_ylabel('vel z (m)')
        axs[2, 1].set_xlabel('time (s)')
        plt.show()
        # Plot in 3D.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(pos_ref_traj[:, 0], pos_ref_traj[:, 1], pos_ref_traj[:, 2])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.show()
