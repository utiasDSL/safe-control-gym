'''1D, 2D, and 3D quadrotor environment using PyBullet physics.

Based on UTIAS Dynamic Systems Lab's gym-pybullet-drones:
    * https://github.com/utiasDSL/gym-pybullet-drones
'''

import math
from copy import deepcopy

import casadi as cs
from gym import spaces
import numpy as np
import pybullet as p

from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.envs.gym_pybullet_drones.base_aviary import BaseAviary
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType, cmd2pwm, pwm2rpm
from safe_control_gym.math_and_models.transformations import transform_trajectory, csRotXYZ

class Quadrotor(BaseAviary):
    '''1D, 2D, and 3D quadrotor environment task.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.
    '''

    NAME = 'quadrotor'
    AVAILABLE_CONSTRAINTS = deepcopy(GENERAL_CONSTRAINTS)

    DISTURBANCE_MODES = { # Set at runtime by QUAD_TYPE
        'observation': {
            'dim': -1
        },
        'action': {
            'dim': -1
        },
        'dynamics': {
            'dim': -1
        }
    }

    INERTIAL_PROP_RAND_INFO = {
        'M': { # Nominal: 0.027
            'distrib': 'uniform',
            'low': 0.022,
            'high': 0.032
        },
        'Ixx': { # Nominal: 1.4e-5
            'distrib': 'uniform',
            'low': 1.3e-5,
            'high': 1.5e-5
        },
        'Iyy': { # Nominal: 1.4e-5
            'distrib': 'uniform',
            'low': 1.3e-5,
            'high': 1.5e-5
        },
        'Izz': { # Nominal: 2.17e-5
            'distrib': 'uniform',
            'low': 2.07e-5,
            'high': 2.27e-5
        }
    }

    INIT_STATE_RAND_INFO = {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_y': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_y_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_z': {
            'distrib': 'uniform',
            'low': 0.1,
            'high': 1.5
        },
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_phi': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_psi': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_p': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_theta_dot': {  # TODO: replace with q.
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_q': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_r': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        }
    }

    TASK_INFO = {
        'stabilization_goal': [0, 1],
        'stabilization_goal_tolerance': 0.05,
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [0.5, 0],
        'trajectory_scale': -0.5,
        'proj_point': [0, 0, 0.5],
        'proj_normal': [0, 1, 1],
    }

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 # custom args
                 quad_type: QuadType = QuadType.TWO_D,
                 norm_act_scale=0.1,
                 obs_goal_horizon=0,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 done_on_out_of_bound=True,
                 info_mse_metric_state_weight=None,
                 **kwargs
                 ):
        '''Initialize a quadrotor environment.

        Args:
            init_state (ndarray, optional): The initial state of the environment, (z, z_dot) or (x, x_dot, z, z_dot theta, theta_dot).
            inertial_prop (ndarray, optional): The inertial properties of the environment (M, Ixx, Iyy, Izz).
            quad_type (QuadType, optional): The choice of motion type (1D along z, 2D in the x-z plane, or 3D).
            norm_act_scale (float): Scaling the [-1,1] action space around hover thrust when `normalized_action_space` is True.
            obs_goal_horizon (int): How many future goal states to append to obervation.
            rew_state_weight (list/ndarray): Quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): Quadratic weights for action in rl reward.
            rew_exponential (bool): If to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): If to termiante when state is out of bound.
            info_mse_metric_state_weight (list/ndarray): Quadratic weights for state in mse calculation for info dict.
        '''

        self.QUAD_TYPE = QuadType(quad_type)
        self.norm_act_scale = norm_act_scale
        self.obs_goal_horizon = obs_goal_horizon
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.rew_exponential = rew_exponential
        self.done_on_out_of_bound = done_on_out_of_bound
        if info_mse_metric_state_weight is None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.info_mse_metric_state_weight = np.array([1,0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.info_mse_metric_state_weight = np.array([1,0,1,0,0,0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.THREE_D:
                self.info_mse_metric_state_weight = np.array([1,0,1,0,1,0,0,0,0,0,0,0], ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), not implemented quad type.')
        else:
            if (self.QUAD_TYPE == QuadType.ONE_D and len(info_mse_metric_state_weight)==2) or \
                    (self.QUAD_TYPE == QuadType.TWO_D and len(info_mse_metric_state_weight)==6) or \
                    (self.QUAD_TYPE == QuadType.THREE_D and len(info_mse_metric_state_weight)==12):
                self.info_mse_metric_state_weight = np.array(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), wrong info_mse_metric_state_weight argument size.')

        # BaseAviary constructor, called after defining the custom args,
        # since some BenchmarkEnv init setup can be task(custom args)-dependent.
        super().__init__(init_state=init_state, inertial_prop=inertial_prop, **kwargs)

        # Store initial state info.
        self.INIT_STATE_LABELS = {
            QuadType.ONE_D: ['init_x', 'init_x_dot'],
            QuadType.TWO_D: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta', 'init_theta_dot'],
            QuadType.THREE_D: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                               'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q',  'init_r']
        }
        if init_state is None:
            for init_name in self.INIT_STATE_RAND_INFO: # Default zero state.
                self.__dict__[init_name.upper()] = 0.
        else:
            if isinstance(init_state, np.ndarray):  # Full state as numpy array .
                for i, init_name in enumerate(self.INIT_STATE_LABELS[self.QUAD_TYPE]):
                    self.__dict__[init_name.upper()] = init_state[i]
            elif isinstance(init_state, dict):  # Partial state as dictionary.
                for init_name in self.INIT_STATE_LABELS[self.QUAD_TYPE]:
                    self.__dict__[init_name.upper()] = init_state.get(init_name, 0.)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), init_state incorrect format.')

        # Remove randomization info of initial state components inconsistent with quad type.
        for init_name in list(self.INIT_STATE_RAND_INFO.keys()):
            if init_name not in self.INIT_STATE_LABELS[self.QUAD_TYPE]:
                self.INIT_STATE_RAND_INFO.pop(init_name, None)
        # Remove randomization info of inertial components inconsistent with quad type.
        if self.QUAD_TYPE == QuadType.ONE_D:
            # Do NOT randomize J for the 1D quadrotor.
            self.INERTIAL_PROP_RAND_INFO.pop('Ixx', None)
            self.INERTIAL_PROP_RAND_INFO.pop('Iyy', None)
            self.INERTIAL_PROP_RAND_INFO.pop('Izz', None)
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # Only randomize Iyy for the 2D quadrotor.
            self.INERTIAL_PROP_RAND_INFO.pop('Ixx', None)
            self.INERTIAL_PROP_RAND_INFO.pop('Izz', None)

        # Override inertial properties of passed as arguments.
        if inertial_prop is None:
            pass
        elif self.QUAD_TYPE == QuadType.ONE_D and np.array(inertial_prop).shape == (1,):
            self.MASS = inertial_prop[0]
        elif self.QUAD_TYPE == QuadType.TWO_D and np.array(inertial_prop).shape == (2,):
            self.MASS, self.J[1, 1] = inertial_prop
        elif self.QUAD_TYPE == QuadType.THREE_D and np.array(inertial_prop).shape == (4,):
            self.MASS, self.J[0, 0], self.J[1, 1], self.J[2, 2] = inertial_prop
        elif isinstance(inertial_prop, dict):
            self.MASS = inertial_prop.get('M', self.MASS)
            self.J[0, 0] = inertial_prop.get('Ixx', self.J[0, 0])
            self.J[1, 1] = inertial_prop.get('Iyy', self.J[1, 1])
            self.J[2, 2] = inertial_prop.get('Izz', self.J[2, 2])
        else:
            raise ValueError('[ERROR] in Quadrotor.__init__(), inertial_prop incorrect format.')

        # Set prior/symbolic info.
        self._setup_symbolic()

        # Create X_GOAL and U_GOAL references for the assigned task.
        self.U_GOAL = np.ones(self.action_dim) * self.MASS * self.GRAVITY_ACC / self.action_dim
        if self.TASK == Task.STABILIZATION:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.hstack(
                    [self.TASK_INFO['stabilization_goal'][1],
                     0.0])  # x = {z, z_dot}.
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, z, z_dot, theta, theta_dot}.
            elif self.QUAD_TYPE == QuadType.THREE_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0,
                    self.TASK_INFO['stabilization_goal'][2], 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        elif self.TASK == Task.TRAJ_TRACKING:
            POS_REF, VEL_REF, _ = self._generate_trajectory(traj_type=self.TASK_INFO['trajectory_type'],
                                              traj_length=self.EPISODE_LEN_SEC,
                                              num_cycles=self.TASK_INFO['num_cycles'],
                                              traj_plane=self.TASK_INFO['trajectory_plane'],
                                              position_offset=self.TASK_INFO['trajectory_position_offset'],
                                              scaling=self.TASK_INFO['trajectory_scale'],
                                              sample_time=self.CTRL_TIMESTEP
                                              ) # Each of the 3 returned values is of shape (Ctrl timesteps, 3)
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 2], # z
                    VEL_REF[:, 2]  # z_dot
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0], # x
                    VEL_REF[:, 0], # x_dot
                    POS_REF[:, 2], # z
                    VEL_REF[:, 2], # z_dot
                    np.zeros(POS_REF.shape[0]), # zeros
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.THREE_D:
                # Additional transformation of the originally planar trajectory.
                POS_REF_TRANS, VEL_REF_TRANS = transform_trajectory(
                    POS_REF, VEL_REF, trans_info={
                        'point': self.TASK_INFO['proj_point'],
                        'normal': self.TASK_INFO['proj_normal'],
                    })
                self.X_GOAL = np.vstack([
                    POS_REF_TRANS[:, 0], # x
                    VEL_REF_TRANS[:, 0], # x_dot
                    POS_REF_TRANS[:, 1], # y
                    VEL_REF_TRANS[:, 1], # y_dot
                    POS_REF_TRANS[:, 2], # z
                    VEL_REF_TRANS[:, 2], # z_dot
                    np.zeros(POS_REF_TRANS.shape[0]), # zeros
                    np.zeros(POS_REF_TRANS.shape[0]),
                    np.zeros(POS_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0])
                ]).transpose()

        # Equilibrium point at hover for linearization.
        self.X_EQ = np.zeros(self.state_dim)
        self.U_EQ = self.U_GOAL


    def reset(self):
        '''(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Returns:
            obs (ndarray): The initial state of the environment.
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''

        super().before_reset()
        # PyBullet simulation reset.
        super()._reset_simulation()

        # Choose randomized or deterministic inertial properties.
        prop_values = {
            'M': self.MASS,
            'Ixx': self.J[0, 0],
            'Iyy': self.J[1, 1],
            'Izz': self.J[2, 2]
        }
        if self.RANDOMIZED_INERTIAL_PROP:
            prop_values = self._randomize_values_by_info(
                prop_values, self.INERTIAL_PROP_RAND_INFO)
            if any(phy_quantity < 0 for phy_quantity in prop_values.values()):
                raise ValueError('[ERROR] in Quadrotor.reset(), negative randomized inertial properties.')
        self.OVERRIDDEN_QUAD_MASS = prop_values['M']
        self.OVERRIDDEN_QUAD_INERTIA = [prop_values['Ixx'], prop_values['Iyy'], prop_values['Izz']]

        # Override inertial properties.
        p.changeDynamics(
            self.DRONE_IDS[0],
            linkIndex=-1,  # Base link.
            mass=self.OVERRIDDEN_QUAD_MASS,
            localInertiaDiagonal=self.OVERRIDDEN_QUAD_INERTIA,
            physicsClientId=self.PYB_CLIENT)

        # Randomize initial state.
        init_values = {init_name: self.__dict__[init_name.upper()]
                       for init_name in self.INIT_STATE_LABELS[self.QUAD_TYPE]}
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)
        INIT_XYZ = [init_values.get('init_'+k, 0.) for k in ['x', 'y', 'z']]
        INIT_VEL = [init_values.get('init_'+k+'_dot', 0.) for k in ['x', 'y', 'z']]
        INIT_RPY = [init_values.get('init_'+k, 0.) for k in ['phi', 'theta', 'psi']]
        if self.QUAD_TYPE == QuadType.TWO_D:
            INIT_ANG_VEL = [0, init_values.get('init_theta_dot', 0.), 0]
        else:
            INIT_ANG_VEL = [init_values.get('init_'+k, 0.) for k in ['p', 'q', 'r']]  # TODO: transform from body rates.
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], INIT_XYZ,
                                          p.getQuaternionFromEuler(INIT_RPY),
                                          physicsClientId=self.PYB_CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[0], INIT_VEL, INIT_ANG_VEL,
                            physicsClientId=self.PYB_CLIENT)

        # Update BaseAviary internal variables before calling self._get_observation().
        self._update_and_store_kinematic_information()
        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)

        # Return either an observation and dictionary or just the observation.
        if self.INFO_IN_RESET:
            return obs, info
        else:
            return obs


    def step(self, action):
        '''Advances the environment by one control step.

        Pass the commanded RPMs and the adversarial force to the superclass .step().
        The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times in BaseAviary.

        Args:
            action (ndarray): The action applied to the environment for the step.

        Returns:
            obs (ndarray): The state of the environment after the step.
            reward (float): The scalar reward/cost of the step.
            done (bool): Whether the conditions for the end of an episode are met in the step.
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''

        # Get the preprocessed rpm for each motor
        rpm = super().before_step(action)

        # Determine disturbance force.
        disturb_force = None
        passive_disturb = 'dynamics' in self.disturbances
        adv_disturb = self.adversary_disturbance == 'dynamics'
        if passive_disturb or adv_disturb:
            disturb_force = np.zeros(self.DISTURBANCE_MODES['dynamics']['dim'])
        if passive_disturb:
            disturb_force = self.disturbances['dynamics'].apply(
                disturb_force, self)
        if adv_disturb and self.adv_action is not None:
            disturb_force = disturb_force + self.adv_action
            # Clear the adversary action, wait for the next one.
            self.adv_action = None
        # Construct full (3D) disturbance force.
        if disturb_force is not None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                # Only disturb on z direction.
                disturb_force = [0, 0, float(disturb_force)]
            elif self.QUAD_TYPE == QuadType.TWO_D:
                # Only disturb on x-z plane.
                disturb_force = [float(disturb_force[0]), 0, float(disturb_force[1])]
            elif self.QUAD_TYPE == QuadType.THREE_D:
                disturb_force = np.asarray(disturb_force).flatten()

        # Advance the simulation.
        super()._advance_simulation(rpm, disturb_force)
        # Standard Gym return.
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info

    def render(self, mode='human'):
        '''Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            frame (ndarray): A multidimensional array with the RGB frame captured by PyBullet's camera.
        '''

        [w, h, rgb, _, _] = p.getCameraImage(width=self.RENDER_WIDTH,
                                                 height=self.RENDER_HEIGHT,
                                                 shadow=1,
                                                 viewMatrix=self.CAM_VIEW,
                                                 projectionMatrix=self.CAM_PRO,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                 physicsClientId=self.PYB_CLIENT)
        # Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA').show()
        return np.reshape(rgb, (h, w, 4))

    def _setup_symbolic(self):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost. '''
        m, g, l = self.MASS, self.GRAVITY_ACC, self.L
        Iyy = self.J[1, 1]
        dt = self.CTRL_TIMESTEP
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        if self.QUAD_TYPE == QuadType.ONE_D:
            nx, nu = 2, 1
            # Define states.
            X = cs.vertcat(z, z_dot)
            # Define input thrust.
            T = cs.MX.sym('T')
            U = cs.vertcat(T)
            # Define dynamics equations.
            X_dot = cs.vertcat(z_dot, T / m - g)
            # Define observation equation.
            Y = cs.vertcat(z, z_dot)
        elif self.QUAD_TYPE == QuadType.TWO_D:
            nx, nu = 6, 2
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            theta = cs.MX.sym('theta')
            theta_dot = cs.MX.sym('theta_dot')
            X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
            # Define input thrusts.
            T1 = cs.MX.sym('T1')
            T2 = cs.MX.sym('T2')
            U = cs.vertcat(T1, T2)
            # Define dynamics equations.
            X_dot = cs.vertcat(x_dot,
                               cs.sin(theta) * (T1 + T2) / m, z_dot,
                               cs.cos(theta) * (T1 + T2) / m - g, theta_dot,
                               l * (T2 - T1) / Iyy / np.sqrt(2))
            # Define observation.
            Y = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        elif self.QUAD_TYPE == QuadType.THREE_D:
            nx, nu = 12, 4
            Ixx = self.J[0, 0]
            Izz = self.J[2, 2]
            J = cs.blockcat([[Ixx, 0.0, 0.0],
                             [0.0, Iyy, 0.0],
                             [0.0, 0.0, Izz]])
            Jinv = cs.blockcat([[1.0/Ixx, 0.0, 0.0],
                                [0.0, 1.0/Iyy, 0.0],
                                [0.0, 0.0, 1.0/Izz]])
            gamma = self.KM/self.KF
            x = cs.MX.sym('x')
            y = cs.MX.sym('y')
            phi = cs.MX.sym('phi')  # Roll
            theta = cs.MX.sym('theta')  # Pitch
            psi = cs.MX.sym('psi')  # Yaw
            x_dot = cs.MX.sym('x_dot')
            y_dot = cs.MX.sym('y_dot')
            p_body = cs.MX.sym('p')  # Body frame roll rate
            q_body = cs.MX.sym('q')  # body frame pith rate
            r_body = cs.MX.sym('r')  # body frame yaw rate
            # PyBullet Euler angles use the SDFormat for rotation matrices.
            Rob = csRotXYZ(phi, theta, psi)  # rotation matrix transforming a vector in the body frame to the world frame.

            # Define state variables.
            X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)

            # Define inputs.
            f1 = cs.MX.sym('f1')
            f2 = cs.MX.sym('f2')
            f3 = cs.MX.sym('f3')
            f4 = cs.MX.sym('f4')
            U = cs.vertcat(f1, f2, f3, f4)

            # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. 'Design of a trajectory tracking controller for a
            # nanoquadcopter.' arXiv preprint arXiv:1608.05786 (2016).

            # Defining the dynamics function.
            # We are using the velocity of the base wrt to the world frame expressed in the world frame.
            # Note that the reference expresses this in the body frame.
            oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1+f2+f3+f4)/m - cs.vertcat(0, 0, g)
            pos_ddot = oVdot_cg_o
            pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
            Mb = cs.vertcat(l/cs.sqrt(2.0)*(f1+f2-f3-f4),
                            l/cs.sqrt(2.0)*(-f1+f2+f3-f4),
                            gamma*(-f1+f2-f3+f4))
            rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p_body, q_body, r_body)) @ J @ cs.vertcat(p_body, q_body, r_body)))
            ang_dot = cs.blockcat([[1, cs.sin(phi)*cs.tan(theta), cs.cos(phi)*cs.tan(theta)],
                                   [0, cs.cos(phi), -cs.sin(phi)],
                                   [0, cs.sin(phi)/cs.cos(theta), cs.cos(phi)/cs.cos(theta)]]) @ cs.vertcat(p_body, q_body, r_body)
            X_dot = cs.vertcat(pos_dot[0], pos_ddot[0], pos_dot[1], pos_ddot[1], pos_dot[2], pos_ddot[2], ang_dot, rate_dot)

            Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {
            'cost_func': cost_func,
            'vars': {
                'X': X,
                'U': U,
                'Xr': Xr,
                'Ur': Ur,
                'Q': Q,
                'R': R
            }
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)

    def _set_action_space(self):
        '''Sets the action space of the environment. '''
        # Define action/input dimension, labels, and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            action_dim = 1
            self.ACTION_LABELS = ['T']
            self.ACTION_UNITS = ['N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            action_dim = 2
            self.ACTION_LABELS = ['T1', 'T2']
            self.ACTION_UNITS = ['N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        elif self.QUAD_TYPE == QuadType.THREE_D:
            action_dim = 4
            self.ACTION_LABELS = ['T1', 'T2', 'T3', 'T4']
            self.ACTION_UNITS = ['N', 'N', 'N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-', '-']

        n_mot = 4 / action_dim
        a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
        a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
        self.physical_action_bounds = (np.full(action_dim, a_low, np.float32),
                                       np.full(action_dim, a_high, np.float32))

        if self.NORMALIZED_RL_ACTION_SPACE:
            # Normalized thrust (around hover thrust).
            self.hover_thrust = self.GRAVITY_ACC * self.MASS / action_dim
            self.action_space = spaces.Box(low=-np.ones(action_dim),
                                           high=np.ones(action_dim),
                                           dtype=np.float32)
        else:
            # Direct thrust control.
            self.action_space = spaces.Box(low=self.physical_action_bounds[0],
                                            high=self.physical_action_bounds[1],
                                            dtype=np.float32)

    def _set_observation_space(self):
        '''Sets the observation space of the environment. '''
        self.x_threshold = 2
        self.y_threshold = 2
        self.z_threshold = 2
        self.phi_threshold_radians = 85 * math.pi / 180
        self.theta_threshold_radians = 85 * math.pi / 180
        self.psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        # Define obs/state bounds, labels and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            # obs/state = {z, z_dot}.
            low = np.array([self.GROUND_PLANE_Z, -np.finfo(np.float32).max])
            high = np.array([self.z_threshold, np.finfo(np.float32).max])
            self.STATE_LABELS = ['z', 'z_dot']
            self.STATE_UNITS = ['m', 'm/s']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # obs/state = {x, x_dot, z, z_dot, theta, theta_dot}.
            low = np.array([
                -self.x_threshold, -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
                -self.theta_threshold_radians, -np.finfo(np.float32).max
            ])
            high = np.array([
                self.x_threshold, np.finfo(np.float32).max,
                self.z_threshold, np.finfo(np.float32).max,
                self.theta_threshold_radians, np.finfo(np.float32).max
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'rad', 'rad/s']
        elif self.QUAD_TYPE == QuadType.THREE_D:
            # obs/state = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            low = np.array([
                -self.x_threshold, -np.finfo(np.float32).max,
                -self.y_threshold, -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
                -self.phi_threshold_radians, -self.theta_threshold_radians, -self.psi_threshold_radians,
                -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max
            ])
            high = np.array([
                self.x_threshold, np.finfo(np.float32).max,
                self.y_threshold, np.finfo(np.float32).max,
                self.z_threshold, np.finfo(np.float32).max,
                self.phi_threshold_radians, self.theta_threshold_radians, self.psi_threshold_radians,
                np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                 'phi', 'theta', 'psi', 'p', 'q', 'r']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Concatenate reference for RL.
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            # Include future goal state(s).
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            low = np.concatenate([low] * mul)
            high = np.concatenate([high] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            low = np.concatenate([low] * 2)
            high = np.concatenate([high] * 2)

        # Define obs space exposed to the controller.
        # Note how the obs space can differ from state space (i.e. augmented with the next reference states for RL)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _setup_disturbances(self):
        '''Sets up the disturbances. '''
        # Custom disturbance info.
        self.DISTURBANCE_MODES['observation']['dim'] = self.obs_dim
        self.DISTURBANCE_MODES['action']['dim'] = self.action_dim
        self.DISTURBANCE_MODES['dynamics']['dim'] = int(self.QUAD_TYPE)
        super()._setup_disturbances()

    def _preprocess_control(self, action):
        '''Converts the action passed to .step() into motors' RPMs (ndarray of shape (4,)).

        Args:
            action (ndarray): The raw action input, of size 1 or 2 depending on QUAD_TYPE.

        Returns:
            action (ndarray): The motors RPMs to apply to the quadrotor.
        '''

        if self.NORMALIZED_RL_ACTION_SPACE:
            # rescale action to around hover thrust
            action = (1 + self.norm_act_scale * action) * self.hover_thrust
        self.current_physical_action = action

        # Apply disturbances.
        if 'action' in self.disturbances:
            action = self.disturbances['action'].apply(action, self)
        if self.adversary_disturbance == 'action':
            action = action + self.adv_action
        self.current_noisy_physical_action = action

        thrust = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        self.current_clipped_action = thrust

        # convert to quad motor rpm commands
        pwm = cmd2pwm(thrust, self.PWM2RPM_SCALE, self.PWM2RPM_CONST, self.KF, self.MIN_PWM, self.MAX_PWM)
        rpm = pwm2rpm(pwm, self.PWM2RPM_SCALE, self.PWM2RPM_CONST)
        return rpm

    def _get_observation(self):
        '''Returns the current observation (state) of the environment.

        Returns:
            obs (ndarray): The state of the quadrotor, of size 2 or 6 depending on QUAD_TYPE.
        '''

        full_state = self._get_drone_state_vector(0)
        pos, _, rpy, vel, ang_v, _ = np.split(full_state, [3, 7, 10, 13, 16])
        if self.QUAD_TYPE == QuadType.ONE_D:
            # {z, z_dot}.
            self.state = np.hstack([pos[2], vel[2]]).reshape((2,))
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # {x, x_dot, z, z_dot, theta, theta_dot}.
            self.state = np.hstack(
                [pos[0], vel[0], pos[2], vel[2], rpy[1], ang_v[1]]
            ).reshape((6,))
        elif self.QUAD_TYPE == QuadType.THREE_D:
            Rob = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape((3,3))
            Rbo = Rob.T
            ang_v_body_frame = Rbo @ ang_v
            # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            self.state = np.hstack(
                # [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]  # Note: world ang_v != body frame pqr
                [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v_body_frame]
            ).reshape((12,))
        # if not np.array_equal(self.state,
        #                       np.clip(self.state, self.observation_space.low, self.observation_space.high)):
        #     if self.GUI and self.VERBOSE:
        #         print(
        #             '[WARNING]: observation was clipped in Quadrotor._get_observation().'
        #         )

        # Apply observation disturbance.
        obs = deepcopy(self.state)
        if 'observation' in self.disturbances:
            obs = self.disturbances['observation'].apply(obs, self)

        # Concatenate goal info (references state(s)) for RL.
        obs = self.extend_obs(obs, self.ctrl_step_counter+1)
        return obs

    def _get_reward(self):
        '''Computes the current step's reward value.

        Returns:
            reward (float): The evaluated reward/cost.
        '''
        # RL cost.
        if self.COST == Cost.RL_REWARD:
            state = self.state
            act = np.asarray(self.current_noisy_physical_action)
            act_error = act - self.U_GOAL
            # Quadratic costs w.r.t state and action
            # TODO: consider using multiple future goal states for cost in tracking
            if self.TASK == Task.STABILIZATION:
                state_error = state - self.X_GOAL
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act_error * act_error)
            if self.TASK == Task.TRAJ_TRACKING:
                wp_idx = min(self.ctrl_step_counter, self.X_GOAL.shape[0]-1)
                state_error = state - self.X_GOAL[wp_idx]
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act_error * act_error)
            rew = -dist
            # Convert rew to be positive and bounded [0,1].
            if self.rew_exponential:
                rew = np.exp(rew)
            return rew

        # Control cost.
        if self.COST == Cost.QUADRATIC:
            if self.TASK == Task.STABILIZATION:
                return float(-1 * self.symbolic.loss(x=self.state,
                                                     Xr=self.X_GOAL,
                                                     u=self.current_clipped_action,
                                                     Ur=self.U_GOAL,
                                                     Q=self.Q,
                                                     R=self.R)['l'])
            if self.TASK == Task.TRAJ_TRACKING:
                return float(-1 * self.symbolic.loss(x=self.state,
                                                     Xr=self.X_GOAL[self.ctrl_step_counter,:],
                                                     u=self.current_clipped_action,
                                                     Ur=self.U_GOAL,
                                                     Q=self.Q,
                                                     R=self.R)['l'])

    def _get_done(self):
        '''Computes the conditions for termination of an episode.

        Returns:
            done (bool): Whether an episode is over.
        '''
        # Done if goal reached for stabilization task with quadratic cost.
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            self.goal_reached = bool(np.linalg.norm(self.state - self.X_GOAL) < self.TASK_INFO['stabilization_goal_tolerance'])
            if self.goal_reached:
                return True

        # # Done if the episode length is exceeded.
        # if (self.ctrl_step_counter + 1) / self.CTRL_FREQ >= self.EPISODE_LEN_SEC:
        #     return True

        # # Done if a constraint is violated.
        # if self.constraints is not None:
        #     if self.DONE_ON_VIOLATION and self.constraints.is_violated(self):
        #         return True

        # Done if state is out-of-bounds.
        if self.done_on_out_of_bound:
            if self.QUAD_TYPE == QuadType.ONE_D:
                mask = np.array([1, 0])
            if self.QUAD_TYPE == QuadType.TWO_D:
                mask = np.array([1, 0, 1, 0, 1, 0])
            if self.QUAD_TYPE == QuadType.THREE_D:
                mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
            # Element-wise or to check out-of-bound conditions.
            self.out_of_bounds = np.logical_or(self.state < self.state_space.low,
                                         self.state > self.state_space.high)
            # Mask out un-included dimensions (i.e. velocities)
            self.out_of_bounds = np.any(self.out_of_bounds * mask)
            # Early terminate if needed.
            if self.out_of_bounds:
                return True
        self.out_of_bounds = False

        return False

    def _get_info(self):
        '''Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''
        info = {}
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        if self.done_on_out_of_bound:
            info['out_of_bounds'] = self.out_of_bounds
        # Add MSE.
        state = deepcopy(self.state)
        if self.TASK == Task.STABILIZATION:
            state_error = state - self.X_GOAL
        elif self.TASK == Task.TRAJ_TRACKING:
            # TODO: should use angle wrapping
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter, self.X_GOAL.shape[0]-1)
            state_error = state - self.X_GOAL[wp_idx]
        # Filter only relevant dimensions.
        state_error = state_error * self.info_mse_metric_state_weight
        info['mse'] = np.sum(state_error ** 2)
        # if self.constraints is not None:
        #     info['constraint_values'] = self.constraints.get_values(self)
        #     info['constraint_violations'] = self.constraints.get_violations(self)
        return info

    def _get_reset_info(self):
        '''Generates the info dictionary returned by every call to .reset().

        Returns:
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''
        info = {}
        info['symbolic_model'] = self.symbolic
        info['physical_parameters'] = {
            'quadrotor_mass': self.MASS,
            'quadrotor_iyy_inertia': self.J[1, 1]
        }
        info['x_reference'] = self.X_GOAL
        info['u_reference'] = self.U_GOAL
        if self.constraints is not None:
            info['symbolic_constraints'] = self.constraints.get_all_symbolic_models()
        return info
