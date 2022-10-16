"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) interStepLearn (optional)
        4) interEpisodeLearn (optional)

"""
import numpy as np

from collections import deque

from competition.custom_utils import update_waypoints_avoid_obstacles

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

import scipy.integrate
from math import sqrt, sin, cos, atan2, log1p, pi
from copy import deepcopy

try:
    import custom_utils as utils
except ImportError:
    # PyTest import.
    from . import custom_utils as utils

#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Kinematic limits
        # TODO determine better estimates from model
        v_max = 1.5
        a_max = 0.5
        j_max = 2
        # Affect curve radius around waypoints, higher value means larger curve, smaller value means tighter curve
        self.grad_scale = .4 * v_max

        self.mass = initial_info['nominal_physical_parameters']['quadrotor_mass']
        self.tall = initial_info["gate_dimensions"]["tall"]["height"]
        self.low = initial_info["gate_dimensions"]["low"]["height"]

        ### Spline fitting between waypoints ###

        self.length = len(self.NOMINAL_GATES) + 2
        # end goal
        waypoints = [[self.initial_obs[0], self.initial_obs[2], self.initial_obs[4], self.initial_obs[8]],]

        self.half_pi = 0.5*pi
        for idx, g in enumerate(self.NOMINAL_GATES):
            [x, y, _, _, _, yaw, typ] = g
            yaw += self.half_pi
            z = self.tall if typ == 0 else self.low
            waypoints.append([x, y, z, yaw])
        # end goal
        waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4], initial_info["x_reference"][8]])

        # "time" for each waypoint
        # time interval determined by eulcidean distance between waypoints along xyz plane
        self.ts = np.zeros(self.length)
        [x_prev, y_prev, z_prev, _] = waypoints[0]
        for idx in range(1,self.length):
            [x_curr, y_curr, z_curr, _] = waypoints[idx]
            norm = sqrt((x_curr-x_prev)**2 + (y_curr-y_prev)**2 + (z_curr-z_prev)**2)
            self.ts[idx] = self.ts[idx-1] + norm / v_max

        # Flip gates
        # Selectively flip gate orientation based on vector from previous and to next waypoint,
        # and current gate's orientation
        [x_prev, y_prev, _, _] = waypoints[0]
        [x_curr, y_curr, _, yaw_curr] = waypoints[1]
        dt = self.ts[1] - self.ts[0]
        for idx in range(1,self.length-1):
            [x_next, y_next, _, yaw_next] = waypoints[idx+1]
            dt_next = self.ts[idx+1] - self.ts[idx]
            dxf = cos(yaw_curr) * self.grad_scale
            dyf = sin(yaw_curr) * self.grad_scale
            vx1 = np.array((x_curr - x_prev,dt))
            vx21 = np.array((dxf,1))
            vx22 = np.array((-dxf,1))
            vx3 = np.array((x_next - x_curr,dt_next))

            vy1 = np.array((y_curr - y_prev,dt))
            vy21 = np.array((dyf,1))
            vy22 = np.array((-dyf,1))
            vy3 = np.array((y_next - y_curr,dt_next))
            choice_1 = vx1.dot(vx21)/np.linalg.norm(vx1)/np.linalg.norm(vx21)+vx21.dot(vx3)/np.linalg.norm(vx21)/np.linalg.norm(vx3)
            choice_1 += vy1.dot(vy21)/np.linalg.norm(vy1)/np.linalg.norm(vy21)+vy21.dot(vy3)/np.linalg.norm(vy21)/np.linalg.norm(vy3)
            choice_2 = vx1.dot(vx22)/np.linalg.norm(vx1)/np.linalg.norm(vx22)+vx22.dot(vx3)/np.linalg.norm(vx22)/np.linalg.norm(vx3)
            choice_2 += vy1.dot(vy22)/np.linalg.norm(vy1)/np.linalg.norm(vy22)+vy22.dot(vy3)/np.linalg.norm(vy22)/np.linalg.norm(vy3)
            if choice_2 > choice_1:
                waypoints[idx][3] += pi
                print("flipping", idx)
            x_prev, y_prev = x_curr, y_curr
            x_curr, y_curr = x_next, y_next
            yaw_curr = yaw_next
            dt = dt_next

        # Slightly shift gate waypoint to deal with over/undershoot of PID controller
        # E.g. for double left     > shift gate to right
        #      for double right    > shift gate to left
        #      for right then left > shift gate to right
        #      for left then right > shift gate to left
        # For consecutive turns in same direction, distance shifted is larger
        # shift_dist = .2
        # yaw_prev = waypoints[0][3]
        # yaw_curr = waypoints[1][3]
        # dyaw = yaw_curr - yaw_prev
        # for idx in range(1,self.length-1):
        #     yaw_next = waypoints[idx+1][3]
        #     dyaw_next = yaw_next - yaw_curr
        #     scale = (((dyaw + dyaw_next) / 2. / pi + 1.) % 2.) - 1.
        #     if dyaw > 0:
        #         scale = -scale
        #     if dyaw * dyaw_next < 0:
        #         scale = -scale
        #     waypoints[idx][0] -= sin(yaw_curr) * shift_dist * scale
        #     waypoints[idx][1] -= cos(yaw_curr) * shift_dist * scale
        #     yaw_prev = yaw_curr
        #     yaw_curr = yaw_next
        #     dyaw = dyaw_next

        is_collision = True
        if is_collision:
            x_coeffs = np.zeros((6,len(waypoints)-1))
            y_coeffs = np.zeros((6,len(waypoints)-1))
            z_coeffs = np.zeros((6,len(waypoints)-1))
            [x0, y0, z0, yaw0] = waypoints[0]
            dx0 = cos(yaw0)*0.01
            dy0 = sin(yaw0)*0.01
            dz0 = v_max
            dzf = 0
            d2x0 = 0
            d2y0 = 0
            d2z0 = a_max
            d2xf = 0
            d2yf = 0
            d2zf = 0
            for idx in range(1, self.length):
                [xf, yf, zf, yawf] = waypoints[idx]
                dt = self.ts[idx] - self.ts[idx - 1]
                inv_t = 1/dt
                dxf = cos(yawf) * self.grad_scale
                dyf = sin(yawf) * self.grad_scale
                if idx == self.length-1: # yaw of last goal not important
                    dxf = dx0 * 0.01
                    dyf = dy0 * 0.01
                x_coeffs[::-1,idx-1] = utils.quintic_interp(inv_t, x0, xf, dx0, dxf, d2x0, d2xf)
                y_coeffs[::-1,idx-1] = utils.quintic_interp(inv_t, y0, yf, dy0, dyf, d2y0, d2yf)
                z_coeffs[::-1,idx-1] = utils.quintic_interp(inv_t, z0, zf, dz0, dzf, d2z0, d2zf)
                [x0, y0, z0] = [xf, yf, zf]
                [dx0, dy0, dz0] = [dxf, dyf, dzf]
                [d2x0, d2y0, d2z0] = [d2xf, d2yf, d2zf]
            waypoints = np.array(waypoints)
            # z_coeffs = scipy.interpolate.PchipInterpolator(self.ts, waypoints[:,2], 0).c

            # modify endpoint gradient for smooth z ending (pchip iterpolator)
            # z_coeffs[::-1,-1] = utils.cubic_interp(1/(self.ts[-1]-self.ts[-2]),
            #     waypoints[-2,2],
            #     waypoints[-1,2],
            #     z_coeffs[-2,-1],
            #     0
            # )
            self.coeffs = [x_coeffs, y_coeffs, z_coeffs]

            # Integrate to get pathlength
            pathlength = 0
            for idx in range(self.length-1):
                pathlength += scipy.integrate.quad(
                    utils.df_idx(self.length, self.ts, x_coeffs, y_coeffs, z_coeffs),
                    0, self.ts[idx+1] - self.ts[idx])[0]
            self.scaling_factor = self.ts[-1] / pathlength

            duration = self.ts[-1] - self.ts[0]
            t_scaled = np.linspace(self.ts[0], self.ts[-1], int(duration*self.CTRL_FREQ))
            t_diff_scaled = np.zeros(t_scaled.shape)
            gate_scaled = np.zeros(t_scaled.shape, dtype=np.ushort)
            gate = 0
            for i, t in enumerate(t_scaled):
                if gate < self.length and t > self.ts[gate+1]:
                    gate += 1
                t_diff_scaled[i] = t_scaled[i] - self.ts[gate]
                gate_scaled[i] = gate
            x_scaled = np.array(tuple(map(
                lambda idx, t: utils.f(x_coeffs, idx, t),
                gate_scaled, t_diff_scaled)))
            y_scaled = np.array(tuple(map(
                lambda idx, t: utils.f(y_coeffs, idx, t),
                gate_scaled, t_diff_scaled)))
            z_scaled = np.array(tuple(map(
                lambda idx, t: utils.f(z_coeffs, idx, t),
                gate_scaled, t_diff_scaled)))
            spline_waypoints = np.dstack((x_scaled, y_scaled, z_scaled))[0]
            is_collision, waypoints = update_waypoints_avoid_obstacles(spline_waypoints, waypoints, self.NOMINAL_OBSTACLES, 
                                                        initial_info)
            if self.VERBOSE:
                print(x_coeffs)
                print(y_coeffs)
                print(z_coeffs)
                print(waypoints)
                # Plot trajectory in each dimension and 3D.
                plot_trajectory(t_scaled, waypoints, x_scaled, y_scaled, z_scaled)
                # Draw the trajectory on PyBullet's GUI
                draw_trajectory(initial_info, waypoints, x_scaled, y_scaled, z_scaled)

        ### S-curve ###
        sf = pathlength

        s = np.zeros(8)
        v = np.zeros(8)
        a = np.zeros(8)
        j = np.array((j_max, 0, -j_max, 0, -j_max, 0, j_max, 0))
        t = np.zeros(8)
        T = np.zeros(8)

        T[1] = T[3] = T[5] = T[7] = min(sqrt(v_max/j_max), a_max/j_max) # min(wont't hit a_max, will hit a_max)
        utils.fill(T, t, s, v, a, j)
        ds = sf - s[-1]
        if ds < 0:
            # T[2] = T[4] = T[6] = 0
            T[1] = T[3] = T[5] = T[7] = (xf/(2*j_max))**(1./3)
        else:
            T_2_max = (v_max - 2*v[1])/a_max
            added_T_2 = (-v[1]+sqrt(v[1]*v[1]/2 + a_max*ds))/a_max
            if added_T_2 > T_2_max:
                T[2] = T[6] = T_2_max
                ds -= (2*v[1] + a_max*T_2_max)*T_2_max
                T[4] = ds/v_max
            else:
                # T[4] = 0
                T[2] = T[6] = added_T_2
        utils.fill(T, t, s, v, a, j)
        def s_curve(t_):
            for i in range(7):
                if t[i+1] > t_:
                    break
            dt = t_ - t[i]
            s_ = ((j[i]/6*dt + a[i]/2)*dt + v[i])*dt+ s[i]
            v_ =  (j[i]/2*dt + a[i]  )*dt + v[i]
            a_ =   j[i]  *dt + a[i]
            j_ =   j[i]
            return [s_, v_, a_, j_]
        self.s = s_curve
        self.start_t = -1
        self.end_t = t[-1]
        self.scaled_t = 0
        self.time_scale = 1
        self.takeoff_land_duration = 2
        self.at_gate = False
        self.gate_no = 0
        self.run_coeffs = deepcopy(self.coeffs)
        self.run_ts = np.copy(self.ts)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if self.scaled_t < self.end_t:
            if info and info['current_target_gate_in_range']:
                if not self.at_gate:
                    # Local replan when near to goal
                    self.at_gate = True
                    [x, y, z, _, _, yaw] = info['current_target_gate_pos']
                    yaw += self.half_pi
                    dt = self.run_ts[self.gate_no + 1] - self.curve_t
                    inv_t = 1/dt
                    dx_gate = cos(yaw) * self.grad_scale
                    dy_gate = sin(yaw) * self.grad_scale
                    dz_gate = 0
                    for i, (val, dval) in enumerate(zip([x,y,z], [dx_gate, dy_gate, dz_gate])):
                        self.run_coeffs[i] = np.insert(self.run_coeffs[i], self.gate_no+1, np.flip(utils.quintic_interp(
                            inv_t,
                            self.target_pose[i],
                            val,
                            self.tangent[i],
                            dval,
                            self.dtangent[i],
                            0
                        )), axis=1)
                    dt = self.run_ts[self.gate_no + 2] - self.run_ts[self.gate_no + 1]
                    inv_t = 1/dt
                    for i, (val, dval) in enumerate(zip([x,y,z], [dx_gate, dy_gate, dz_gate])):
                        self.run_coeffs[i][::-1,self.gate_no+2] = utils.quintic_interp(
                            inv_t,
                            val,
                            utils.f(self.run_coeffs[i], self.gate_no+2, dt),
                            dval,
                            utils.df(self.run_coeffs[i], self.gate_no+2, dt),
                            0,
                            0
                        )
                    self.run_ts = np.insert(self.run_ts, self.gate_no+1, self.curve_t)
                    self.length += 1
                    self.gate_no += 1
            else:
                self.at_gate = False

            self.scaled_t += self.time_scale / self.CTRL_FREQ / self.scaling_factor

            [curve_t, curve_v, curve_a, curve_j] = self.s(self.scaled_t)
            curve_t *= self.scaling_factor
            curve_v *= self.scaling_factor * self.time_scale
            curve_a *= self.scaling_factor * self.time_scale
            curve_j *= self.scaling_factor * self.time_scale
            self.curve_t = curve_t

            if (self.gate_no < self.length-2 and self.run_ts[self.gate_no+1] <= curve_t):
                self.gate_no += 1

            t = curve_t - self.run_ts[self.gate_no]

            self.target_pose = np.array([utils.f(coeffs, self.gate_no, t) for coeffs in self.run_coeffs])
            self.tangent = np.array([utils.df(coeffs, self.gate_no, t) for coeffs in self.run_coeffs])
            self.dtangent = np.array([utils.d2f(coeffs, self.gate_no, t) for coeffs in self.run_coeffs])
            self.d2tangent = np.array([utils.d3f(coeffs, self.gate_no, t) for coeffs in self.run_coeffs])
            
            error = np.array((obs[0] - self.target_pose[0], obs[2] - self.target_pose[1], obs[4] - self.target_pose[2])) # positional error (world frame)
            # print(error)
            self.time_scale += -log1p(np.linalg.norm(error)-.25)*0.05
            self.time_scale = max(0.0, min(1.0, self.time_scale))
            # print(self.time_scale)

            target_yaw = atan2(self.tangent[1], self.tangent[0])
            target_vel = self.tangent * curve_v
            target_acc = self.tangent * curve_a + self.dtangent * curve_v**2
            
            # Roll, pitch rate
            # Small angle approximation
            target_jerk = self.tangent * curve_j + self.d2tangent * curve_v**3 + 3 * self.dtangent * curve_v * curve_a
            Jinv = utils.rpy2rot(obs[6], obs[7], obs[8]).transpose()
            body_jerk = np.matmul(Jinv, target_jerk.transpose())
            p = self.mass/9.8*body_jerk[0,1]  #  roll rate = mass / g * y_jerk
            q = self.mass/9.8*body_jerk[0,0]  # pitch rate = mass / g * x_jerk

            # Yaw rate
            den = np.linalg.norm(self.tangent[:2])
            if den < 1e-9:
                r = 0
            else:
                num = self.tangent[0] * self.dtangent[1] - self.tangent[1] * self.dtangent[0]
                r = num/den
                r *= curve_v

            target_rpy_rates = np.array((p,q,r))
            # target_vel = np.zeros(3)
            # target_acc = np.zeros(3)
            # target_yaw = 0.
            # target_rpy_rates = np.zeros(3)
            self.args = [self.target_pose, target_vel, target_acc, target_yaw, target_rpy_rates]
        else:
            self.args[1] = np.zeros(3)
            self.args[2] = np.zeros(3)
            self.args[4] = np.zeros(3)
        command_type = Command(1)  # cmdFullState.
        args = self.args

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the IROS 2022 Safe Robot Learning competition.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    @timing_step
    def interStepLearn(self,
                       action,
                       obs,
                       reward,
                       done,
                       info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1

        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        # Reset run variables
        self.scaled_t = 0
        self.at_gate = False
        self.gate_no = 0
        self.run_coeffs = deepcopy(self.coeffs)
        self.run_ts = np.copy(self.ts)
        self.length = len(self.NOMINAL_GATES) + 2

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
