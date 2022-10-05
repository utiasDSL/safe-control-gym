"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 4 blocks starting with
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
import scipy.interpolate
import scipy.integrate
from math import sqrt, sin, cos, atan2, pi

from collections import deque

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # Test import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory


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
        # Save environment and conrol parameters.
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
            # Initialize a simple PID Controller ror debugging and test
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
        v_max = 2
        a_max = .2
        j_max = .2
        # Affect curve radius around waypoints, higher value means larger curve, smaller value means tighter curve
        grad_scale = 1

        ### Spline fitting between waypoints ###

        length = len(self.NOMINAL_GATES) + 2
        # end goal
        if use_firmware:
            waypoints = [[self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"], self.initial_obs[8]],]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [[self.initial_obs[0], self.initial_obs[2], self.initial_obs[4], self.initial_obs[8]],]

        half_pi = 0.5*pi
        for idx, g in enumerate(self.NOMINAL_GATES):
            [x, y, _, _, _, yaw, typ] = g
            yaw += half_pi
            z = initial_info["gate_dimensions"]["tall"]["height"] if typ == 0 else initial_info["gate_dimensions"]["low"]["height"]
            waypoints.append([x, y, z, yaw])
        # end goal
        waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4], initial_info["x_reference"][8]])

        x_coeffs = np.zeros((4,len(waypoints)-1))
        y_coeffs = np.zeros((4,len(waypoints)-1))

        # "time" for each waypoint
        # time interval determined by eulcidean distance between waypoints along xy plane
        # ts = np.arange(length)
        ts = np.zeros(length)
        [x_prev, y_prev, _, _] = waypoints[0]
        for idx in range(1,length):
            [x_curr, y_curr, _, _] = waypoints[idx]
            xy_norm = sqrt((x_curr-x_prev)**2 + (y_curr-y_prev)**2)
            ts[idx] = ts[idx-1] + xy_norm / v_max

        # Flip gates
        # Selectively flip gate orientation based on vector from previous and to next waypoint,
        # and current gate's orientation
        [x_prev, y_prev, _, _] = waypoints[0]
        [x_curr, y_curr, _, yaw_curr] = waypoints[1]
        dt = ts[1] - ts[0]
        for idx in range(1,length-1):
            [x_next, y_next, _, yaw_next] = waypoints[idx+1]
            dt_next = ts[idx+1] - ts[idx]
            dxf = cos(yaw_curr) * grad_scale
            dyf = sin(yaw_curr) * grad_scale
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
        shift_dist = .1
        yaw_prev = waypoints[0][3]
        yaw_curr = waypoints[1][3]
        dyaw = yaw_curr - yaw_prev
        for idx in range(1,length-1):
            yaw_next = waypoints[idx+1][3]
            dyaw_next = yaw_next - yaw_curr
            scale = (((abs(dyaw) + abs(dyaw_next)) / 2. / pi + 1.) % 2.) - 1.
            if dyaw > 0:
                scale = -scale
            if dyaw * dyaw_next < 0:
                scale = -scale
            waypoints[idx][0] -= sin(yaw_curr) * shift_dist * scale
            waypoints[idx][1] -= cos(yaw_curr) * shift_dist * scale
            yaw_prev = yaw_curr
            yaw_curr = yaw_next
            dyaw = dyaw_next

        [x0, y0, _, yaw0] = waypoints[0]
        dx0 = cos(yaw0)
        dy0 = sin(yaw0)
        for idx in range(1, length):
            [xf, yf, _, yawf] = waypoints[idx]
            dt = ts[idx] - ts[idx - 1]
            inv_t = 1/dt
            inv_t2 = inv_t*inv_t
            dxf = cos(yawf) * grad_scale
            dyf = sin(yawf) * grad_scale
            dx = xf - x0
            dy = yf - y0
            x_a0 = x0
            x_a1 = dx0
            x_a2 = (3*dx*inv_t - 2*dx0 - dxf)*inv_t
            x_a3 = (-2*dx*inv_t + (dxf + dx0))*inv_t2
            x_coeffs[:,idx-1] = (x_a3, x_a2, x_a1, x_a0)
            y_a0 = y0
            y_a1 = dy0
            y_a2 = (3*dy*inv_t - 2*dy0 - dyf)*inv_t
            y_a3 = (-2*dy*inv_t + (dyf + dy0))*inv_t2
            y_coeffs[:,idx-1] = (y_a3, y_a2, y_a1, y_a0)
            [x0, y0] = [xf, yf]
            [dx0, dy0] = [dxf, dyf]
        waypoints = np.array(waypoints)
        z_coeffs = scipy.interpolate.PchipInterpolator(ts, waypoints[:,2], 0).c

        def df_idx(idx):
            def df(t):
                dx = (3*x_coeffs[1,idx]*t + 2*x_coeffs[2,idx])*t + x_coeffs[3,idx]
                dy = (3*y_coeffs[1,idx]*t + 2*y_coeffs[2,idx])*t + y_coeffs[3,idx]
                dz = (3*z_coeffs[1,idx]*t + 2*z_coeffs[2,idx])*t + z_coeffs[3,idx]
                return sqrt(dx*dx+dy*dy+dz*dz)
            return df

        def f_(coeffs):
            def f(t):
                for idx in range(length-1):
                    if (ts[idx+1] > t):
                        break
                t -= ts[idx]
                return ((coeffs[0,idx]*t + coeffs[1,idx])*t + coeffs[2,idx])*t + coeffs[3,idx]
            return f

        def df_(coeffs):
            def f(t):
                for idx in range(length-1):
                    if (ts[idx+1] > t):
                        break
                t -= ts[idx]
                return (3*coeffs[0,idx]*t + 2*coeffs[1,idx])*t + coeffs[2,idx]
            return f

        def d2f_(coeffs):
            def f(t):
                for idx in range(length-1):
                    if (ts[idx+1] > t):
                        break
                t -= ts[idx]
                return 6*coeffs[0,idx]*t + 2*coeffs[1,idx]
            return f

        def d3f_(coeffs):
            def f(t):
                for idx in range(length-1):
                    if (ts[idx+1] > t):
                        break
                t -= ts[idx]
                return 6*coeffs[0,idx]
            return f

        # Integrate to get pathlength
        pathlength = 0
        for idx in range(length-1):
            pathlength += scipy.integrate.quad(df_idx(idx), 0, ts[idx+1] - ts[idx])[0]
        self.scaling_factor = ts[-1] / pathlength

        self.x = f_(x_coeffs)
        self.y = f_(y_coeffs)
        self.z = f_(z_coeffs)

        self.dx = df_(x_coeffs)
        self.dy = df_(y_coeffs)
        self.dz = df_(z_coeffs)

        self.d2x = d2f_(x_coeffs)
        self.d2y = d2f_(y_coeffs)
        self.d2z = d2f_(z_coeffs)

        self.d3x = d3f_(x_coeffs)
        self.d3y = d3f_(y_coeffs)
        self.d3z = d3f_(z_coeffs)

        duration = ts[-1] - ts[0]
        t_scaled = np.linspace(ts[0], ts[-1], int(duration*self.CTRL_FREQ))
        x_scaled = np.array(tuple(map(self.x, t_scaled)))
        y_scaled = np.array(tuple(map(self.y, t_scaled)))
        z_scaled = np.array(tuple(map(self.z, t_scaled)))

        if self.VERBOSE:
            print(x_coeffs)
            print(y_coeffs)
            print(z_coeffs)
            print(waypoints)
            print(t_scaled)
            print(x_scaled)
            print(y_scaled)
            print(z_scaled)
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

        def fill():
            for i in range(7):
                dt = T[i+1]
                t[i+1] = t[i] + dt
                s[i+1] = ((j[i]/6*dt + a[i]/2)*dt + v[i])*dt+ s[i]
                v[i+1] =  (j[i]/2*dt + a[i]  )*dt + v[i]
                a[i+1] =   j[i]  *dt + a[i]

        T[1] = T[3] = T[5] = T[7] = min(sqrt(v_max/j_max), a_max/j_max) # min(wont't hit a_max, will hit a_max)
        fill()
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
        fill()
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
        def rpy2rot(roll, pitch, yaw):
            sr = sin(roll)
            cr = cos(roll)
            sp = sin(pitch)
            cp = cos(pitch)
            sy = sin(yaw)
            cy = cos(yaw)
            m = np.matrix(((cy*cp, -sy*cr+cy*sp*sr,  sy*sr+cy*cr*sp),
                        (sy*cp,  cy*cr+sy*sp*sr, -cy*sr+sp*sy*cr),
                        (-sp   ,  cp*sr         ,  cp*cr)))
            return m
        self.rpy2rot = rpy2rot


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

        t = time / self.scaling_factor

        # # Handwritten solution for GitHub's getting_stated scenario.
        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif iteration >= 3*self.CTRL_FREQ:
            if self.start_t < 0:
                self.start_t = t
            t -= self.start_t
            if t > self.end_t:
                height = 0.
                duration = 3
                command_type = Command(3)  # Land.
                args = [height, duration]
            [curve_t, curve_v, curve_a, curve_j] = self.s(self.scaled_t)
            curve_t *= self.scaling_factor
            curve_v *= self.scaling_factor
            curve_a *= self.scaling_factor
            curve_j *= self.scaling_factor
            x_c = self.x(curve_t)
            y_c = self.y(curve_t)
            z_c = self.z(curve_t)
            target_pos = np.array([x_c, y_c, z_c])
            # print(obs[0] - x_c, obs[2] - y_c, obs[4] - z_c)

            dx_c = self.dx(curve_t)
            dy_c = self.dy(curve_t)
            dz_c = self.dz(curve_t)
            d2x_c = self.d2x(curve_t)
            d2y_c = self.d2y(curve_t)
            d2z_c = self.d2z(curve_t)
            d3x_c = self.d3x(curve_t)
            d3y_c = self.d3y(curve_t)
            d3z_c = self.d3z(curve_t)
            tangent = np.array((dx_c,dy_c,dz_c))
            dtangent = np.array((d2x_c,d2y_c,d2z_c))
            d2tangent = np.array((d3x_c,d3y_c,d3z_c))

            target_yaw = atan2(dy_c, dx_c)
            target_vel = tangent * curve_v
            target_acc = tangent * curve_a + dtangent * curve_v**2
            
            # Roll, pitch rate
            # Small angle approximation
            target_jerk = tangent * curve_j + d2tangent * curve_v**3 + 3 * dtangent * curve_v * curve_a
            Jinv = self.rpy2rot(obs[6], obs[7], obs[8]).transpose()
            body_jerk = np.matmul(Jinv, target_jerk.transpose())
            p = 0.027/9.8*body_jerk[0,1]  #  roll rate = mass / g * y_jerk
            q = 0.027/9.8*body_jerk[0,0]  # pitch rate = mass / g * x_jerk

            # Yaw rate
            den = np.linalg.norm(tangent[:2])
            if den < 1e-9:
                r = 0
            else:
                num = dx_c * d2y_c - dy_c * d2x_c
                r = num/den
                r *= curve_v

            target_rpy_rates = np.array((p,q,r))
            # target_vel = np.zeros(3)
            # target_acc = np.zeros(3)
            # target_yaw = 0.
            # target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
            

        # elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
        #     step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
        #     target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
        #     target_vel = np.zeros(3)
        #     target_acc = np.zeros(3)
        #     target_yaw = 0.
        #     target_rpy_rates = np.zeros(3)

        #     command_type = Command(1)  # cmdFullState.
        #     args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        # elif iteration == 20*self.CTRL_FREQ:
        #     command_type = Command(6)  # notify setpoint stop.
        #     args = []

        # elif iteration == 20*self.CTRL_FREQ+1:
        #     x = self.ref_x[-1]
        #     y = self.ref_y[-1]
        #     z = 1.5 
        #     yaw = 0.
        #     duration = 2.5

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif iteration == 23*self.CTRL_FREQ:
        #     x = self.initial_obs[0]
        #     y = self.initial_obs[2]
        #     z = 1.5
        #     yaw = 0.
        #     duration = 6

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif iteration == 30*self.CTRL_FREQ:
        #     height = 0.
        #     duration = 3

        #     command_type = Command(3)  # Land.
        #     args = [height, duration]

        # elif iteration == 33*self.CTRL_FREQ-1:
        #     command_type = Command(-1)  # Terminate command to be sent once trajectory is completed.
        #     args = []

        else:
            command_type = Command(0)  # None.
            args = []

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
