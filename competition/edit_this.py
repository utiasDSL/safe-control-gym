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
from math import sqrt, sin, cos, atan2, pi, log1p

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
        v_max = 1.5
        a_max = .5
        j_max = .5
        self.mass = initial_info['nominal_physical_parameters']['quadrotor_mass']
        # Affect curve radius around waypoints, higher value means larger curve, smaller value means tighter curve
        self.grad_scale = .8

        self.tall = initial_info["gate_dimensions"]["tall"]["height"]
        self.low = initial_info["gate_dimensions"]["low"]["height"]

        ### Spline fitting between waypoints ###

        self.length = len(self.NOMINAL_GATES) + 2
        # end goal
        if use_firmware:
            waypoints = [[self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"], self.initial_obs[8]],]  # Height is hardcoded scenario knowledge.
        else:
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
        # time interval determined by eulcidean distance between waypoints along xy plane
        self.ts = np.zeros(self.length)
        [x_prev, y_prev, _, _] = waypoints[0]
        for idx in range(1,self.length):
            [x_curr, y_curr, _, _] = waypoints[idx]
            xy_norm = sqrt((x_curr-x_prev)**2 + (y_curr-y_prev)**2)
            self.ts[idx] = self.ts[idx-1] + xy_norm / v_max

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

        self.cubic_interp = lambda inv_t, x0, xf, dx0, dxf:(
            x0,
            dx0,
            (3*(xf-x0)*inv_t - 2*dx0 - dxf)*inv_t,
            (-2*(xf-x0)*inv_t + (dxf + dx0))*inv_t*inv_t)

        self.quintic_interp = lambda inv_t, x0, xf, dx0, dxf, d2x0, d2xf:(
            x0,
            dx0,
            d2x0*0.5,
            ((10*(xf-x0)*inv_t - (4*dxf+6*dx0))*inv_t - 0.5*(3*d2x0-d2xf))*inv_t,
            ((-15*(xf-x0)*inv_t + (7*dxf+8*dx0))*inv_t + 0.5*(3*d2x0-2*d2xf))*inv_t*inv_t,
            ((6*(xf-x0)*inv_t - 3*(dxf+dx0))*inv_t - 0.5*(d2x0-d2xf))*inv_t*inv_t*inv_t)

        self.x_coeffs = np.zeros((6,len(waypoints)-1))
        self.y_coeffs = np.zeros((6,len(waypoints)-1))
        [x0, y0, _, yaw0] = waypoints[0]
        dx0 = cos(yaw0)
        dy0 = sin(yaw0)
        d2x0 = 0
        d2y0 = 0
        d2xf = 0
        d2yf = 0
        for idx in range(1, self.length):
            [xf, yf, _, yawf] = waypoints[idx]
            dt = self.ts[idx] - self.ts[idx - 1]
            inv_t = 1/dt
            dxf = cos(yawf) * self.grad_scale
            dyf = sin(yawf) * self.grad_scale
            if idx == 0 or idx == self.length-1: # don't need to care about curving out of first and into last goal
                dxf *= 0.01
                dyf *= 0.01
            self.x_coeffs[::-1,idx-1] = self.quintic_interp(inv_t, x0, xf, dx0, dxf, d2x0, d2xf)
            self.y_coeffs[::-1,idx-1] = self.quintic_interp(inv_t, y0, yf, dy0, dyf, d2y0, d2yf)
            [x0, y0] = [xf, yf]
            [dx0, dy0] = [dxf, dyf]
            # [d2x0, d2y0] = [d2xf, d2yf]
        waypoints = np.array(waypoints)
        self.z_coeffs = scipy.interpolate.PchipInterpolator(self.ts, waypoints[:,2], 0).c

        def f_(coeffs):
            def f(idx, t):
                val = 0
                for coeff in coeffs[:,idx]:
                    val *= t
                    val += coeff
                return val
            return f

        def df_(coeffs):
            def f(idx, t):
                coeffs_no = len(coeffs[:,idx])
                val = 0
                for i, coeff in enumerate(coeffs[:-1,idx]):
                    val *= t
                    factor = (coeffs_no - i - 1)
                    val += coeff * factor
                return val
            return f

        def d2f_(coeffs):
            def f(idx, t):
                coeffs_no = len(coeffs[:,idx])
                val = 0
                for i, coeff in enumerate(coeffs[:-2,idx]):
                    val *= t
                    factor = (coeffs_no - i - 1)
                    val += coeff * factor * (factor - 1)
                return val
            return f

        def d3f_(coeffs):
            def f(idx, t):
                coeffs_no = len(coeffs[:,idx])
                val = 0
                for i, coeff in enumerate(coeffs[:-3,idx]):
                    val *= t
                    factor = (coeffs_no - i - 1)
                    val += coeff * factor * (factor - 1) * (factor - 2)
                return val
            return f

        def df_idx():
            def df(t):
                for idx in range(self.length-1):
                    if (self.ts[idx+1] > t):
                        break
                t -= self.ts[idx]
                dx = self.df(self.x_coeffs)(idx, t)
                dy = self.df(self.y_coeffs)(idx, t)
                dz = self.df(self.z_coeffs)(idx, t)
                return sqrt(dx*dx+dy*dy+dz*dz)
            return df

        self.f = f_
        self.df = df_
        self.d2f = d2f_
        self.d3f = d3f_

        # Integrate to get pathlength
        pathlength = 0
        for idx in range(self.length-1):
            pathlength += scipy.integrate.quad(df_idx(), 0, self.ts[idx+1] - self.ts[idx])[0]
        self.scaling_factor = self.ts[-1] / pathlength

        if self.VERBOSE:
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
            x_scaled = np.array(tuple(map(self.f(self.x_coeffs), gate_scaled, t_diff_scaled)))
            y_scaled = np.array(tuple(map(self.f(self.y_coeffs), gate_scaled, t_diff_scaled)))
            z_scaled = np.array(tuple(map(self.f(self.z_coeffs), gate_scaled, t_diff_scaled)))
            print(self.x_coeffs)
            print(self.y_coeffs)
            print(self.z_coeffs)
            print(waypoints)
            print(t_scaled)
            print(t_diff_scaled)
            print(gate_scaled)
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
        self.scaled_t = 0
        self.time_scale = 1
        self.takeoff_land_duration = 2

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
        self.at_gate = False
        self.gate_no = 0
        self.run_x_coeffs = np.copy(self.x_coeffs)
        self.run_y_coeffs = np.copy(self.y_coeffs)
        self.run_z_coeffs = np.copy(self.z_coeffs)
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

        # # Handwritten solution for GitHub's getting_stated scenario.
        if iteration == 0:
            height = 1
            duration = self.takeoff_land_duration

            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif iteration >= self.takeoff_land_duration*self.CTRL_FREQ and self.scaled_t < self.end_t:
            if info['current_target_gate_in_range']:
                if not self.at_gate:
                    # Local replan when near to goal
                    self.at_gate = True
                    [x, y, z, _, _, yaw] = info['current_target_gate_pos']
                    yaw += self.half_pi
                    dt = self.run_ts[self.gate_no + 1] - self.curve_t
                    inv_t = 1/dt
                    dx_gate = cos(yaw) * self.grad_scale
                    dy_gate = sin(yaw) * self.grad_scale
                    self.run_x_coeffs = np.insert(self.run_x_coeffs, self.gate_no+1, np.flip(self.quintic_interp(
                        inv_t,
                        self.x_c,
                        x,
                        self.dx_c,
                        dx_gate,
                        self.d2x_c,
                        0
                    )), axis=1)
                    self.run_y_coeffs = np.insert(self.run_y_coeffs, self.gate_no+1, np.flip(self.quintic_interp(
                        inv_t,
                        self.y_c,
                        y,
                        self.dy_c,
                        dy_gate,
                        self.d2y_c,
                        0
                    )), axis=1)
                    self.run_z_coeffs = np.insert(self.run_z_coeffs, self.gate_no+1, np.flip(self.cubic_interp(
                        inv_t,
                        self.z_c,
                        z,
                        self.dz_c,
                        self.df(self.run_z_coeffs)(self.gate_no, dt)
                    )), axis=1)
                    dt = self.run_ts[self.gate_no + 2] - self.run_ts[self.gate_no + 1]
                    inv_t = 1/dt
                    self.run_x_coeffs[::-1,self.gate_no+2] = self.quintic_interp(
                        inv_t,
                        x,
                        self.f(self.run_x_coeffs)(self.gate_no+2, dt),
                        dx_gate,
                        self.df(self.run_x_coeffs)(self.gate_no+2, dt),
                        0,
                        0
                    )
                    self.run_y_coeffs[::-1,self.gate_no+2] = self.quintic_interp(
                        inv_t,
                        y,
                        self.f(self.run_y_coeffs)(self.gate_no+2, dt),
                        dy_gate,
                        self.df(self.run_y_coeffs)(self.gate_no+2, dt),
                        0,
                        0
                    )
                    self.run_z_coeffs[::-1,self.gate_no+2] = self.cubic_interp(
                        inv_t,
                        z,
                        self.f(self.run_z_coeffs)(self.gate_no+2, dt),
                        self.run_z_coeffs[-2, self.gate_no+2],
                        self.df(self.run_z_coeffs)(self.gate_no+2, dt)
                    )
                    self.run_ts = np.insert(self.run_ts, self.gate_no+1, self.curve_t)
                    self.length += 1
                    self.gate_no += 1
                    # print(np.insert(self.run_ts, self.gate_no+1, self.curve_t))
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

            self.x_c = self.f(self.run_x_coeffs)(self.gate_no, t)
            self.y_c = self.f(self.run_y_coeffs)(self.gate_no, t)
            self.z_c = self.f(self.run_z_coeffs)(self.gate_no, t)
            target_pos = np.array([self.x_c, self.y_c, self.z_c])
            error = np.array((obs[0] - self.x_c, obs[2] - self.y_c, obs[4] - self.z_c)) # positional error (world frame)
            # print(error)
            self.time_scale += -log1p(np.linalg.norm(error)-.25)*0.025
            self.time_scale = min(1.0, self.time_scale)
            # print(self.time_scale)

            self.dx_c = self.df(self.run_x_coeffs)(self.gate_no, t)
            self.dy_c = self.df(self.run_y_coeffs)(self.gate_no, t)
            self.dz_c = self.df(self.run_z_coeffs)(self.gate_no, t)
            self.d2x_c = self.d2f(self.run_x_coeffs)(self.gate_no, t)
            self.d2y_c = self.d2f(self.run_y_coeffs)(self.gate_no, t)
            self.d2z_c = self.d2f(self.run_z_coeffs)(self.gate_no, t)
            self.d3x_c = self.d3f(self.run_x_coeffs)(self.gate_no, t)
            self.d3y_c = self.d3f(self.run_y_coeffs)(self.gate_no, t)
            self.d3z_c = self.d3f(self.run_z_coeffs)(self.gate_no, t)
            tangent = np.array((self.dx_c,self.dy_c,self.dz_c))
            dtangent = np.array((self.d2x_c,self.d2y_c,self.d2z_c))
            d2tangent = np.array((self.d3x_c,self.d3y_c,self.d3z_c))

            target_yaw = atan2(self.dy_c, self.dx_c)
            target_vel = tangent * curve_v
            target_acc = tangent * curve_a + dtangent * curve_v**2
            
            # Roll, pitch rate
            # Small angle approximation
            target_jerk = tangent * curve_j + d2tangent * curve_v**3 + 3 * dtangent * curve_v * curve_a
            Jinv = self.rpy2rot(obs[6], obs[7], obs[8]).transpose()
            body_jerk = np.matmul(Jinv, target_jerk.transpose())
            p = self.mass/9.8*body_jerk[0,1]  #  roll rate = mass / g * y_jerk
            q = self.mass/9.8*body_jerk[0,0]  # pitch rate = mass / g * x_jerk

            # Yaw rate
            den = np.linalg.norm(tangent[:2])
            if den < 1e-9:
                r = 0
            else:
                num = self.dx_c * self.d2y_c - self.dy_c * self.d2x_c
                r = num/den
                r *= curve_v

            target_rpy_rates = np.array((p,q,r))
            # target_vel = np.zeros(3)
            # target_acc = np.zeros(3)
            # target_yaw = 0.
            # target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == (self.takeoff_land_duration + self.end_t)*self.CTRL_FREQ:
            command_type = Command(6)  # notify setpoint stop.
            args = []

        elif iteration == (self.takeoff_land_duration + self.end_t)*self.CTRL_FREQ + 1:
            height = 0.
            duration = self.takeoff_land_duration

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == (2*self.takeoff_land_duration + self.end_t)*self.CTRL_FREQ + 2:
            command_type = Command(-1)  # Terminate command to be sent once trajectory is completed.
            args = []

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

        self.scaled_t = 0
        self.at_gate = False
        self.gate_no = 0
        self.run_x_coeffs = np.copy(self.x_coeffs)
        self.run_y_coeffs = np.copy(self.y_coeffs)
        self.run_z_coeffs = np.copy(self.z_coeffs)
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
