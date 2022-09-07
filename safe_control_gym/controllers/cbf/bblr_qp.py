"""Barrier Bayesian Linear Regression (BBLR) quadratic programming (QP) safety filter with online learning
   of Lie derivatives

Reference:
    * [Barrier Bayesian Linear Regression: Online Learning of Control Barrier Conditions for Safety-Critical 
       Control of Uncertain Systems](https://proceedings.mlr.press/v168/brunke22a/brunke22a.pdf)

"""

import matplotlib.pyplot as plt
from sys import platform
import numpy as np
import casadi as cs

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.cbf.bblr_qp_utils import cbf_cartpole, linear_function, cartesian_product
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.utils.logging import ExperimentLogger

class BBLR_QP(BaseController):
    def __init__(self,
                 env_func,
                 unsafe_controller=None,
                 # Runner args.
                 deque_size=10,
                 eval_batch_size=1,
                 output_dir="results/temp",
                 **custom_args):
        
        """
        BBLR-QP controller with online learning of Lie derivatives: The CBF's superlevel set defines a positively control 
        invariant set. A QP based on the CBF's Lie derivative with respect to the dynamics filters arbitrary control
        inputs to keep the system inside the CBF's superlevel set. Due to model mismatch, the Lie derivative is also
        incorrect. This approach learns the error in the Lie derivative online using BBLR to satisfy the Lie derivative 
        condition in the QP for the true system.

        Args:
            env_func (gym.Env): Functionalized initialization of the environment.
            unsafe_controller (BaseController): Underlying controller providing (unsafe) control inputs
            deque_size (int): TODO
            eval_batch_size(int): TODO
            output_dir (str): TODO

        """

        # algo specific args
        for k, v in custom_args.items():
            self.__dict__[k] = v

        self.eval_batch_size = eval_batch_size

        self.env = env_func()
        self.step_size = self.env.PYB_FREQ // self.env.CTRL_FREQ
        self.env = RecordEpisodeStatistics(self.env, deque_size)

        self.input_constraints_sym = self.env.constraints.get_input_constraint_symbolic_models()

        self.unsafe_controller = unsafe_controller

        self.model = self.env.symbolic
        self.X = self.model.x_sym
        self.u = self.model.u_sym

        # Check if the dynamics are control affine
        assert self.is_control_affine()

        # Control barrier function
        self.h_func = cbf_cartpole(self.X, self.x_pos_max, self.x_vel_max, self.theta_max, self.theta_dot_max, cbf_scale=self.cbf_scale)
        # Lie derivative with respect to the dynamics
        self.lie_derivative = self.get_lie_derivative()

        self.linear_func = linear_function(self.slope)

        # Logging.
        self.logger = ExperimentLogger(output_dir)

        # Create h_bar(x), the more conservative cbf for pssf h_bar(x) = h(x) - epsilon_norm
        self.h_bar_func = cbf_cartpole(self.X, self.x_pos_max, self.x_vel_max, self.theta_max, self.theta_dot_max, epsilon_norm=self.epsilon_norm, cbf_scale=self.cbf_scale)

        self.h_max = self.cbf_scale

        x = self.X
        b1 = self.h_func(x)
        b2 = self.h_func(x) ** 2
        b3 = self.h_func(x) ** 3
        b4 = self.h_func(x) ** 4
        b5 = self.h_func(x) ** 5

        b1_lip = 1
        b2_lip = 2 * self.h_max
        b3_lip = 3 * self.h_max ** 2
        b4_lip = 4 * self.h_max ** 3
        b5_lip = 5 * self.h_max ** 4

        self.alpha = None
        self.beta = None
        self.alpha_kappa_bound_constants = np.array([[b1_lip, b2_lip, b3_lip, b4_lip, b5_lip]])
        self.beta_kappa_bound_constants = np.array([[b1_lip * self.u_inf_norm, b2_lip * self.u_inf_norm, b3_lip * self.u_inf_norm, b4_lip * self.u_inf_norm, b5_lip * self.u_inf_norm]])
        basis = cs.vertcat(b1, b2, b3, b4, b5)
        basis_func = cs.Function('basis', [x], [basis], ['x'], ['basis'])

        # Basis
        self.basis = basis_func

        # Initialize posterior weight distributions
        self.num_basis = np.shape(self.basis(0))[0] * 2
        self.w_cov_posterior = np.eye(self.num_basis) * self.w_prior_var
        self.w_mean_posterior = np.zeros((self.num_basis, 1))

    def get_phi_x(self, x):
        """Helper method for BBLR. Gets set of basis functions.
        
        """
        phi_x1 = np.reshape(self.basis(x.reshape(-1, 1)), (-1, 1))
        phi_x2 = np.reshape(self.basis(x.reshape(-1, 1)), (-1, 1))
        return phi_x1, phi_x2

    def update_weight_posterior(self, x, u, y):
        """Helper method for BBLR. Updates the weight posterior.
        
        """
        # Reference https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec19-slides.pdf
        # x: state
        # u: input
        # y: Lie derivative error
        num_data = np.shape(x)[0]

        # Construct basis matrix (num_data by num_basis)
        Phi = np.zeros((num_data, self.num_basis))
        for i in range(num_data):
            phi_x1, phi_x2 = self.get_phi_x(x[i])
            phi_x = np.vstack((phi_x1, u[i] * phi_x2))
            Phi[i] = np.reshape(phi_x, (1, -1))

        # Each row of x and y correspond to a data point
        self.w_cov_posterior = np.linalg.inv(1 / self.noise_var * Phi.transpose() @ Phi + 1 / self.w_prior_var * np.eye(self.num_basis))
        self.w_mean_posterior = 1 / self.noise_var * self.w_cov_posterior @ Phi.transpose() @ y.reshape(-1,1)

    def get_prediction_posterior(self, x):
        """Helper method for BBLR. Obtains the prediction posterior.
        
        """
        # Each row of x is a data point
        num_pts = np.shape(x)[0]
        self.alpha_x_var_posterior = np.zeros((num_pts, 1))
        self.alpha_x_mean_posterior = np.zeros((num_pts, 1))

        self.beta_x_var_posterior = np.zeros((num_pts, 1))
        self.beta_x_mean_posterior = np.zeros((num_pts, 1))

        split_idx = int(self.num_basis/2)
        w_cov_posterior1 = self.w_cov_posterior[:split_idx, :split_idx]
        w_cov_posterior2 = self.w_cov_posterior[split_idx:, split_idx:]

        w_mean_posterior1 = self.w_mean_posterior[:split_idx]
        w_mean_posterior2 = self.w_mean_posterior[split_idx:]

        for i in range(num_pts):
            phi_x1, phi_x2 = self.get_phi_x(x[i])

            self.alpha_x_var_posterior[i] = np.transpose(phi_x1) @ w_cov_posterior1 @ phi_x1 + self.noise_var
            self.alpha_x_mean_posterior[i] = w_mean_posterior1.reshape(1, -1) @ phi_x1

            self.beta_x_var_posterior[i] = np.transpose(phi_x2) @ w_cov_posterior2 @ phi_x2 + self.noise_var
            self.beta_x_mean_posterior[i] = w_mean_posterior2.reshape(1, -1) @ phi_x2

        return self.alpha_x_mean_posterior, self.beta_x_mean_posterior, self.alpha_x_var_posterior, self.beta_x_var_posterior

    def reset(self):
        """Resets the environment

        """
        self.env.reset()

    def get_lie_derivative(self):
        """Determines the Lie derivative of the CBF with respect to the known dynamics

        """
        dVdx = cs.gradient(self.h_func(X=self.X)['cbf'], self.X)
        LfV = cs.dot(dVdx, self.model.x_dot)
        LfV_func = cs.Function('LfV', [self.X, self.u], [LfV], ['X', 'u'], ['LfV'])
        return LfV_func

    def is_control_affine(self):
        """Check if the system is control affine

        """
        dfdu = cs.jacobian(self.model.x_dot, self.u)
        return not cs.depends_on(dfdu, self.u)

    def is_cbf(self, num_points=100, tolerance=0.0):
        """
        Check if the provided CBF candidate (h_bar) is actually a CBF for the system using a gridded approach

        Args:
            num_points (int): The minimum number of points to check for the verification
            tolerance (float): The amount by which the box that contains the grid is extended in every dimension

        Returns:
            valid_cbf (bool): Whether the provided CBF candidate is valid
            infeasible_states (list): List of all states for which the QP is infeasible

        """
        valid_cbf = False

        # Select the states to check the CBF condition
        max_bounds = np.array([self.x_pos_max, self.x_vel_max, self.theta_max, self.theta_dot_max])
        # Add some tolerance to the bounds to also check the condition outside of the superlevel set
        max_bounds += tolerance
        min_bounds = - max_bounds

        # state dimension and input dimension
        nx, nu = self.model.nx, self.model.nu

        # Make sure that every vertex is checked
        num_points = max(2 * nx, num_points + num_points % (2 * nx))
        num_points_per_dim = num_points // nx

        # Create the lists of states to check
        states_to_sample = [np.linspace(min_bounds[i], max_bounds[i], num_points_per_dim) for i in range(nx)]
        states_to_check = cartesian_product(*states_to_sample)

        # Set dummy control input
        control_input = np.ones((nu, 1))

        num_infeasible = 0
        infeasible_states = []

        # Check if the optimization problem is feasible for every considered state
        for state in states_to_check:
            # Certify action without using any learned model
            safe_control_input, success = self.certify_action(state, control_input, use_learned_model=False)
            if not success:
                infeasible_states.append(state)
                num_infeasible += 1

        num_infeasible_states_inside_set = 0

        # Check if the infeasible point is inside or outside the superlevel set. Note that the sampled region makes up a
        # box, but the superlevel set is not. The superlevel set only needs to be contained inside the box.
        
        for infeasible_state in infeasible_states:
            barrier_at_x = self.h_func(X=infeasible_state)['cbf']
            #barrier_at_x = self.h_bar_func(X=infeasible_state)['cbf']
            if barrier_at_x < 0:
                # print("Outside superlevel set:", infeasible_state)
                pass
            else:
                print("Infeasible state inside superlevel set:", infeasible_state)
                num_infeasible_states_inside_set += 1

        print("Number of infeasible states:", num_infeasible)
        print("Number of infeasible states inside superlevel set:", num_infeasible_states_inside_set)

        if num_infeasible_states_inside_set > 0:
            valid_cbf = False
            # print("The provided CBF candidate is not a valid CBF.")
        elif num_infeasible > 0:
            valid_cbf = True
            # print("The provided CBF candidate is a valid CBF inside its superlevel set for the checked states. "
            #       "Consider increasing the sampling resolution to get a more precise evaluation. "
            #       "The CBF is not valid on the entire provided domain. Consider softening the CBF constraint by "
            #       "setting 'soft_constraint: True' inside the config.")
        else:
            valid_cbf = True
            # print("The provided CBF candidate is a valid CBF for the checked states. "
            #       "Consider increasing the sampling resolution to get a more precise evaluation.")

        return valid_cbf, infeasible_states

    def certify_action(self, current_state, unsafe_action, alpha=None, beta=None, use_learned_model=True):
        """Calculates certified control input.

        Args:
            current_state (np.array): current state of the continuous-time system
            unsafe_action (np.array): unsafe control input.
            alpha (numpy.float64): BBLR variable to learn Lie derivative 
            beta (numpy.float64):  BBLR variable to learn Lie derivative
            use_learned_model (bool): Whether the learned Lie derivative is used in the certification

        Returns:
            u_val (np.array): certified control input
            success (bool): Whether the certification was successful

        """

        nx, nu = self.model.nx, self.model.nu

        # Define optimizer and variables
        opti = cs.Opti("conic")

        # optimization variable: control input
        u_var = opti.variable(nu, 1)

        # evaluate at Lie derivative and CBF at the current state
        lie_derivative_at_x = self.lie_derivative(X=current_state, u=u_var)['LfV']
        barrier_at_x = self.h_func(X=current_state)['cbf']

        delta_lie_der = 0.0
        slack = 0.0

        if alpha == None or beta == None:
            # not using learning
            use_learned_model = False

        if use_learned_model:
            delta_lie_der = alpha + beta * u_var
            kappa_bound_constants = np.hstack((self.alpha_kappa_bound_constants, self.beta_kappa_bound_constants))
            kappa_bound_coeff = kappa_bound_constants @ np.abs(self.w_mean_posterior)
            bblr_kappa_bound = kappa_bound_coeff * (barrier_at_x)
        else:
            delta_lie_der = 0
            bblr_kappa_bound = 0 

        if self.soft_constrained:
            slack_var = opti.variable(1, 1)

            # quadratic objective
            cost = 0.5 * cs.norm_2(unsafe_action - u_var) ** 2 + self.slack_weight * slack_var**2

            # soften CBF constraint
            slack = slack_var

            # Non-negativity constraint on slack variable
            opti.subject_to(slack_var >= 0.0)
        else:
            # quadratic objective
            cost = 0.5 * cs.norm_2(unsafe_action - u_var) ** 2

        # CBF constraint
        if use_learned_model:
            # use bblr cbf condition
            assert(self.epsilon_norm <= -self.linear_func(-self.epsilon_norm))
            opti.subject_to(lie_derivative_at_x + delta_lie_der - self.epsilon_norm + slack >= -self.linear_func(barrier_at_x - self.epsilon_norm) + self.linear_func(-1 * self.epsilon_norm) - bblr_kappa_bound)
        else:
            # use standard cbf condition
            opti.subject_to(-self.linear_func(barrier_at_x - self.epsilon_norm) - lie_derivative_at_x <= slack)

        # input constraints
        for input_constraint in self.input_constraints_sym:
            opti.subject_to(input_constraint(u_var) <= 0)

        opti.minimize(cost)

        # set verbosity option of optimizer
        
        # opts = {'printLevel': 'none'}
        # opts = {}
        # select QP solver
        # opti.solver('qpoases', opts)
        if platform == "linux":
            opts = {'printLevel': 'low', 'error_on_fail': False}
            opti.solver('qpoases', opts)
        elif platform == "darwin":
            opts = {'error_on_fail': False}
            opti.solver('qrqp', opts)

        self.opti_dict = {
            "opti": opti,
            "u_var": u_var,
            "cost": cost
        }

        if self.soft_constrained:
            self.opti_dict["slack_var"] = slack_var

        success = False
        try:
            # solve optimization problem
            sol = opti.solve()
            success = True

            u_val = sol.value(u_var)
            if self.soft_constrained:
                slack_var = sol.value(slack_var)
                if slack_var > self.slack_tolerance:
                    print("Slack:", slack_var)
                    success = False
            cost = sol.value(cost)
        except RuntimeError as e:
            print(e)
            success = False
            u_val = opti.debug.value(u_var)
            print("u", u_val)
            if self.soft_constrained:
                slack_val = opti.debug.value(slack_var)
                print("slack", slack_val)
            print(self.lie_derivative(X=current_state, u=u_val)['LfV'])
            print(self.linear_func(x=barrier_at_x)["y"])
            print("------------------------------------------------")

        return u_val, success

    def select_action(self, current_state, alpha=None, beta=None, use_learned_model=True):
        """Select the action to apply to the system.

        """
        if self.unsafe_controller is not None:
            # use underlying (potentially unsafe) control input
            unsafe_input = self.unsafe_controller.select_action(current_state)
        else:
            # create random control input
            unsafe_input = 2.0 * (2.0 * np.random.random(size=self.model.nu) - 1.0)

            # create sinusoidal control input
            # unsafe_input = 0.5 * np.sin(2 * np.pi / 50 * (self.env.pyb_step_counter // self.step_size) - np.pi) + 0.0

        # certify control input
        safe_input, success = self.certify_action(current_state, unsafe_input, alpha, beta, use_learned_model)

        return safe_input, unsafe_input, success

    def run(self, render=False, logging=False):
        """Runs evaluation with current policy, while learning Lie derivative online.

        Args:
            render (bool): if to render during the runs.
            logging (bool): if to log using logger during the runs.

        Returns:
            stats_buffer (CBFBuffer): Buffer of the experiment results

        """
        # load model from training episode
        self.load(self.checkpoint_path)

        counter = 0

        obs = self.env.reset()

        self.logger.add_scalars({
            "x_pos": obs[0],
            "x_dot": obs[1],
            "theta": obs[2],
            "theta_dot": obs[3]},
            counter,
            prefix="state")

        # Initializations for helper variables and arrays for BBLR
        states = np.zeros((self.max_num_steps, self.model.nx))
        inputs = np.zeros((self.max_num_steps, self.model.nu))
        barrier_values = np.zeros((self.max_num_steps, 1))
        lie_derivative_values = np.zeros((self.max_num_steps, 1))
        lie_derivative_est = np.zeros((self.max_num_steps, 1))

        # cbf (or h(x))
        h_stack = np.array([[]])
        h_prior_dot_stack = np.array([[]])

        bblr_std_dev = 0
        alpha = None
        beta = None

        times_learned = 0

        while counter < self.max_num_steps:
            print("Step: ", self.env.pyb_step_counter // self.step_size)

            # BBLR filtering
            safe_action_bblr, unsafe_action, success_bblr = self.select_action(obs, alpha, beta, use_learned_model=self.use_learned_model)

            # standard CBF filtering
            safe_action_standard, _, success_standard = self.select_action(obs, None, None, use_learned_model=False)

            uncertainty_ratio = np.tanh(0.1 * float(bblr_std_dev / (self.epsilon_norm + bblr_std_dev)))
            print("uncertainty ratio:", uncertainty_ratio)
            
            # blend the control inputs
            u_learn_filter = uncertainty_ratio * safe_action_standard + (1 - uncertainty_ratio) * safe_action_bblr

            # Check the system's performance without certification
            if self.use_safe_input:
                action = u_learn_filter
            else:
                action = unsafe_action

            # Step the system
            obs, reward, done, info = self.env.step(action)
            print("obs: {}".format(obs))
            print("action: {}".format(action))

            self.logger.add_scalars({
                "safe_input_h_bar": safe_action_standard,
                "unsafe_input": unsafe_action,
                "applied_input": action},
                counter,
                prefix="action")
            self.logger.add_scalars({
                "x_pos": obs[0],
                "x_dot": obs[1],
                "theta": obs[2],
                "theta_dot": obs[3]},
                counter + 1,
                prefix="state")

            # collect data
            states[counter, :] = obs
            inputs[counter, :] = u_learn_filter;
            barrier_values[counter, :] = self.h_func(X=obs)['cbf']
            lie_derivative_values[counter, :] = self.lie_derivative(X=obs, u=u_learn_filter)['LfV']

            # determine the estiamted lie derivative
            if np.shape(h_stack)[1] > self.n_pts:
                # Compute lie derivative using the forward difference method
                h_dot_stack = np.diff(h_stack) / (1 / self.env.CTRL_FREQ)
                delta_h_dot_stack = h_dot_stack - h_prior_dot_stack[:, 1:]
                
                # Get training data
                state_train = states[times_learned:times_learned + self.n_pts, :][-self.n_pts:]
                input_train = inputs[times_learned:times_learned + self.n_pts, :][-self.n_pts:]
                lie_train = np.transpose(delta_h_dot_stack)[-self.n_pts:]

                self.update_weight_posterior(state_train, input_train, lie_train)

                # Get alpha and beta values
                alpha, beta, alpha_var, beta_var = self.get_prediction_posterior(x=np.transpose(obs))
                alpha = alpha[0][0]; beta = beta[0][0]
                alpha_var = alpha_var[0][0]; beta_var = beta_var[0][0]
                bblr_var = alpha_var + beta_var * safe_action_bblr ** 2
                bblr_std_dev = np.sqrt(bblr_var)
                print("alpha var %.3f beta var %.3f blr var %.3f std dev %.3f\n" % (alpha_var, beta_var, bblr_var, bblr_std_dev))

                times_learned += 1
            else:
                # print a newline to maintain formatting
                print()

            if alpha != None:
                # update our Lie derivative prediction if learning has occured
                lie_derivative_est[counter, :] = lie_derivative_values[counter, :] + alpha + beta * u_learn_filter
            else:
                lie_derivative_est[counter, :] = lie_derivative_values[counter, :]

            h_stack = np.hstack((h_stack, np.array([[float(self.h_func(X=obs)['cbf'])]])))
            h_prior_dot_stack = np.hstack((h_prior_dot_stack, np.array([[float(self.lie_derivative(X=obs, u=u_learn_filter)['LfV'])]])))

            counter += 1
        
        # compare actual and numerical time derivatives to verify online learning
        t = np.arange(self.max_num_steps)
        plt.plot(t[1:-1], h_dot_stack.T, "r", label="h dot numerical")
        plt.plot(t, lie_derivative_values, "b", label="h_dot_hat")
        plt.plot(t, lie_derivative_est, "g", label="h_dot_est")
        plt.xlabel("t")
        plt.ylabel("h_dot")
        plt.legend()
        plt.show()

        return self.logger.stats_buffer