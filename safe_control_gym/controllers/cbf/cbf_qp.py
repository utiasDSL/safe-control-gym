"""Control barrier function (CBF) quadratic programming (QP) safety filter with learned Lie derivatives

Reference:
    * [Learning for Safety-Critical Control with Control Barrier Functions](https://arxiv.org/abs/1912.10099)

"""
import os
from sys import platform
import numpy as np
import casadi as cs
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.cbf.cbf_qp_utils import linear_function, cbf_cartpole, cartesian_product
from safe_control_gym.controllers.cbf.cbf_qp_utils import CBFBuffer
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.math_and_models.neural_networks import MLP
from safe_control_gym.utils.logging import ExperimentLogger


class CBF_QP(BaseController):
    def __init__(self,
                 env_func,
                 unsafe_controller=None,
                 # Runner args.
                 deque_size=10,
                 eval_batch_size=1,
                 output_dir="results/temp",
                 **custom_args):
        """
        CBF-QP controller with learned Lie derivative: The CBF's superlevel set defines a positively control invariant
        set. A QP based on the CBF's Lie derivative with respect to the dynamics allows to filter arbitrary control
        inputs to keep the system inside the CBF's superlevel set. Due to model mismatch, the Lie derivative is also
        incorrect. This approach learns the error in the Lie derivative from multiple experiments to satisfy the
        Lie derivative condition in the QP for the true system.

        Args:
            env_func (gym.Env): Functionalized initialization of the environment.
            unsafe_controller (BaseController): Underlying controller providing (unsafe) control inputs
            deque_size (int): TODO
            eval_batch_size(int): TODO
            output_dir (str): TODO

        """

        # TODO: Combine with CLF approach for stabilization
        # TODO: Currently specific for cartpole! Make more general for other systems, e.g., extend to quadrotor
        #  environment

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

        # TODO: Extend this to systems that are not control affine. Then we would need to move away from a QP solver
        # Check if the dynamics are control affine
        assert self.is_control_affine()

        # Control barrier function
        # TODO: Extend to other systems
        self.cbf = cbf_cartpole(self.X, self.x_pos_max, self.x_vel_max, self.theta_max, self.theta_dot_max)
        # Lie derivative with respect to the dynamics
        self.lie_derivative = self.get_lie_derivative()

        self.linear_func = linear_function(self.slope)
        # TODO: define two different linear functions to be steeper inside the safe set and flatter outside safe set

        # Neural network to learn the residual in the lie derivative
        self.mlp = MLP(self.model.nx, self.model.nu + 1, hidden_dims=self.hidden_dims, activation="relu")
        # optimizer
        self.opt = torch.optim.Adam(self.mlp.parameters(), self.learning_rate)

        max_buffer_size = int(self.max_buffer_size)
        self.buffer = CBFBuffer(self.env.observation_space, self.env.action_space, max_buffer_size,
                                self.train_batch_size)

        # Logging.
        self.logger = ExperimentLogger(output_dir)

    def reset(self):
        """Resets the environment

        """
        self.env.reset()

    def get_lie_derivative(self):
        """Determines the Lie derivative of the CBF with respect to the known dynamics

        """
        dVdx = cs.gradient(self.cbf(X=self.X)['cbf'], self.X)
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
        Check if the provided CBF candidate is actually a CBF for the system using a gridded approach

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
            barrier_at_x = self.cbf(X=infeasible_state)['cbf']
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

    def certify_action(self, current_state, unsafe_action, use_learned_model=True):
        """Calculates certified control input.

        Args:
            current_state (np.array): current state of the continuous-time system
            unsafe_action (np.array): unsafe control input.
            use_learned_model (bool): Whether the learned Lie derivative is used in the certification

        Returns:
            u_val (np.array): certified control input
            success (bool): Whether the certification was successful

        """
        nx, nu = self.model.nx, self.model.nu

        # define optimizer and variables
        opti = cs.Opti("conic")  # Tell casadi that it's a conic problem

        # optimization variable: control input
        u_var = opti.variable(nu, 1)

        # evaluate at Lie derivative and CBF at the current state
        lie_derivative_at_x = self.lie_derivative(X=current_state, u=u_var)['LfV']
        barrier_at_x = self.cbf(X=current_state)['cbf']

        learned_residual = 0.0
        right_hand_side = 0.0

        if use_learned_model:
            torch_state = torch.from_numpy(current_state)
            torch_state = torch.unsqueeze(torch_state, 0)
            torch_state = torch_state.to(torch.float32)
            a_b = self.mlp(torch_state)
            a_b = a_b.detach().numpy()
            a = a_b[0, :self.model.nu]
            b = a_b[0, -1]

            learned_residual = cs.dot(a.T, u_var) + b

        if self.soft_constrained:
            slack_var = opti.variable(1, 1)

            # quadratic objective
            cost = 0.5 * cs.norm_2(unsafe_action - u_var) ** 2 + self.slack_weight * slack_var**2

            # soften CBF constraint
            right_hand_side = slack_var

            # Non-negativity constraint on slack variable
            opti.subject_to(slack_var >= 0.0)
        else:
            # quadratic objective
            cost = 0.5 * cs.norm_2(unsafe_action - u_var) ** 2

        # CBF constraint
        opti.subject_to(-self.linear_func(x=barrier_at_x)["y"] - lie_derivative_at_x - learned_residual <= right_hand_side)

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

    def select_action(self, current_state, use_learned_model=True):
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
        safe_input, success = self.certify_action(current_state, unsafe_input, use_learned_model)

        return safe_input, unsafe_input, success

    def compute_loss(self, batch):
        """Compute training loss of the neural network that represents the Lie derivative error"""
        state, act, barrier_dot, barrier_dot_approx = batch["state"], batch["act"], batch["barrier_dot"], \
                                                      batch["barrier_dot_approx"]

        # predict a and b vectors
        a_b = self.mlp(state)
        a = torch.unsqueeze(a_b[:, 0], 1)
        b = torch.unsqueeze(a_b[:, 1], 1)

        # determine the estimate of h_dot
        h_dot_estimate = barrier_dot + a * act + b

        # determine loss
        loss = (h_dot_estimate - barrier_dot_approx).pow(2).mean()
        return loss

    def update(self, batch):
        """Update the neural network parameters.

        """
        loss = self.compute_loss(batch)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def save(self, path):
        """Saves model params and experiment state to checkpoint path.

        """
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)

        state_dict = {
            "agent": self.mlp.state_dict()
        }
        if self.training:
            exp_state = {
                "buffer": self.buffer.state_dict()
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self, path):
        """Restores model and experiment given checkpoint path.

        """
        state = torch.load(path)

        # restore params
        self.mlp.load_state_dict(state["agent"])

        # restore experiment state
        if self.training:
            self.buffer.load_state_dict(state["buffer"])

    def learn(self):
        """Learn the error in the Lie derivative from multiple experiments.

        """
        input_blending_weight = np.arange(self.num_episodes) / (self.num_episodes - 1)

        # Run experiments in loop
        for i in range(self.num_episodes):
            # reset the episode
            self.reset()

            counter = 0

            obs = self.env.reset()

            # create arrays to collect data
            states = np.zeros((self.max_num_steps, self.model.nx))
            inputs = np.zeros((self.max_num_steps, self.model.nu))
            barrier_values = np.zeros((self.max_num_steps, 1))
            lie_derivative_values = np.zeros((self.max_num_steps, 1))
            lie_derivative_est = np.zeros((self.max_num_steps, 1))

            self.logger.dump_scalars()

            while counter < self.max_num_steps:
                print("Step: ", self.env.pyb_step_counter // self.step_size)

                # determine safe action
                safe_action, unsafe_action, success = self.select_action(obs)

                # blend the safe and unsafe action
                blended_input = (1 - input_blending_weight[i]) * unsafe_action + input_blending_weight[i] * safe_action

                # Step the system
                obs, reward, done, info = self.env.step(blended_input)
                print("obs: {}".format(obs))
                print("action: {}\n".format(safe_action))

                self.logger.add_scalars({
                    "safe_input": safe_action,
                    "unsafe_input": unsafe_action,
                    "applied_input": blended_input},
                    counter,
                    prefix="action")
                self.logger.add_scalars({
                    "x_pos": obs[0],
                    "x_vel": obs[1],
                    "theta": obs[2],
                    "theta_dot": obs[3]},
                    counter,
                    prefix="state")

                # collect data
                states[counter, :] = obs
                inputs[counter, :] = blended_input
                barrier_values[counter, :] = self.cbf(X=obs)['cbf']
                lie_derivative_values[counter, :] = self.lie_derivative(X=obs, u=blended_input)['LfV']

                # Determine the estimated Lie derivative
                torch_state = torch.from_numpy(obs)
                torch_state = torch.unsqueeze(torch_state, 0)
                torch_state = torch_state.to(torch.float32)
                a_b = self.mlp(torch_state)
                a_b = a_b.detach().numpy()
                a = a_b[0, :self.model.nu]
                b = a_b[0, -1]
                lie_derivative_est[counter, :] = lie_derivative_values[counter, :] + np.dot(a.T, blended_input) + b

                counter += 1

            print("Num time steps:", len(inputs))
            print("Certified control input weight:", input_blending_weight[i])

            # numerical time differentiation (symmetric) of barrier function :
            barrier_dot_approx = (barrier_values[2:] - barrier_values[:-2]) / (2 * 1 / self.env.CTRL_FREQ)

            # compare actual and numerical time derivatives
            # import matplotlib.pyplot as plt
            # t = np.arange(self.max_num_steps) / self.env.CTRL_FREQ
            # plt.plot(t[1:-1], barrier_dot_approx, "r", label="h_dot_numerical")
            # plt.plot(t, lie_derivative_values, "b", label="h_dot_hat")
            # plt.plot(t, lie_derivative_est, "g", label="h_dot_est")
            # plt.xlabel("t")
            # plt.ylabel("h_dot")
            # plt.legend()
            # plt.show()

            # Add data to buffer
            self.buffer.push({
                "state": states[1:-1, :],
                "act": inputs[1:-1, :],
                "barrier_dot": lie_derivative_values[1:-1, :],
                "barrier_dot_approx": barrier_dot_approx
            })

            # Update neural network parameters
            for j in range(self.train_iterations):
                batch = self.buffer.sample(self.train_batch_size, self.device)
                self.update(batch)

            # Save model parameters
            print("Saving current model parameters at:", self.checkpoint_path)
            self.save(self.checkpoint_path)

    def run(self,  render=False, logging=False):
        """Runs evaluation with current policy.

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

        # while len(ep_returns) < self.eval_batch_size and counter < self.max_num_steps:
        while counter < self.max_num_steps:
            print("Step: ", self.env.pyb_step_counter // self.step_size)

            safe_action, unsafe_action, success = self.select_action(obs, self.use_learned_model)

            # Check the system's performance without certification
            if self.use_safe_input:
                action = safe_action
            else:
                action = unsafe_action

            obs, reward, done, info = self.env.step(action)
            print("obs: {}".format(obs))
            print("action: {}\n".format(action))

            self.logger.add_scalars({
                "safe_input": safe_action,
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
            counter += 1

        return self.logger.stats_buffer
