'''Control barrier function (CBF) quadratic programming (QP) safety filter with learned Lie derivatives

Reference:
    * [Learning for Safety-Critical Control with Control Barrier Functions](https://arxiv.org/abs/1912.10099)
'''

import os
from typing import Tuple

import torch
import numpy as np
import casadi as cs

from safe_control_gym.safety_filters.cbf.cbf import CBF
from safe_control_gym.safety_filters.cbf.cbf_utils import CBFBuffer
from safe_control_gym.math_and_models.neural_networks import MLP


class CBF_NN(CBF):
    '''Neural Network Based Control Barrier Function Class. '''

    def __init__(self,
                 env_func,
                 # CBF Parameters
                 slope: float=0.1,
                 soft_constrained: bool=True,
                 slack_weight: float=10000.0,
                 slack_tolerance: float=1.0E-3,
                 # NN Parameters
                 max_num_steps: int=250,
                 hidden_dims: list=None,
                 learning_rate: float=0.001,
                 num_episodes: int=20,
                 max_buffer_size: int=1.0E+6,
                 train_batch_size: int=64,
                 train_iterations: int=200,
                 **kwargs):
        '''
        CBF-QP safety filter with learned Lie derivative: The CBF's superlevel set defines a positively control invariant
        set. A QP based on the CBF's Lie derivative with respect to the dynamics allows to filter arbitrary control
        inputs to keep the system inside the CBF's superlevel set. Due to model mismatch, the Lie derivative is also
        incorrect. This approach learns the error in the Lie derivative from multiple experiments to satisfy the
        Lie derivative condition in the QP for the true system.

        Args:
            env_func (BenchmarkEnv): Functionalized initialization of the environment.
            slope (float): The slope of the linear function in the CBF.
            soft_constrainted (bool): Whether to use soft or hard constraints.
            slack_weight (float): The weight of the slack in the optimization.
            slack_tolerance (float): How high the slack can be in the optimization.
            max_num_steps (int): Maximum number of steps to train for.
            hidden_dims (list): Number of hidden dimensions in the neural network.
            learning_rate (float): The learning rate of the neural network.
            num_episodes (int): Number of episodes to train for.
            max_buffer_size (int): Maximum size of the CBFBuffer.
            train_batch_size (int): Size of the batches used during training.
            train_iterations (int): Number of iterations to update policy with batch.
        '''

        super().__init__(env_func, slope, soft_constrained, slack_weight, slack_tolerance, **kwargs)

        self.step_size = self.env.PYB_FREQ // self.env.CTRL_FREQ

        self.max_num_steps = max_num_steps
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.max_buffer_size = max_buffer_size
        self.train_batch_size = train_batch_size
        self.train_iterations = train_iterations

        # Neural network to learn the residual in the lie derivative
        self.mlp = MLP(self.model.nx, self.model.nu + 1, hidden_dims=self.hidden_dims, activation='relu')
        # Optimizer
        self.opt = torch.optim.Adam(self.mlp.parameters(), self.learning_rate)

        max_buffer_size = int(self.max_buffer_size)
        self.buffer = CBFBuffer(self.env.observation_space, self.env.action_space, max_buffer_size, self.device,
                                self.train_batch_size)

        self.uncertified_controller = None
        self.setup_optimizer()

    def setup_optimizer(self):
        '''Setup the certifying CBF problem. '''
        nx, nu = self.model.nx, self.model.nu

        # Define optimizer
        opti = cs.Opti('conic')  # Tell casadi that it's a conic problem

        # Optimization variable: control input
        u_var = opti.variable(nu, 1)

        # States
        current_state_var = opti.parameter(nx, 1)
        uncertified_action_var = opti.parameter(nu, 1)
        a_var = opti.parameter(nu, 1)
        b_var = opti.parameter(1, 1)

        # Evaluate at Lie derivative and CBF at the current state
        lie_derivative_at_x = self.lie_derivative(X=current_state_var, u=u_var)['LfV']
        barrier_at_x = self.cbf(X=current_state_var)['cbf']

        learned_residual = cs.dot(a_var.T, u_var) + b_var

        right_hand_side = 0.0
        slack_var = opti.variable(1, 1)

        if self.soft_constrained:
            # Quadratic objective
            cost = 0.5 * cs.norm_2(uncertified_action_var - u_var) ** 2 + self.slack_weight * slack_var**2

            # Soften CBF constraint
            right_hand_side = slack_var

            # Non-negativity constraint on slack variable
            opti.subject_to(slack_var >= 0.0)
        else:
            opti.subject_to(slack_var == 0.0)
            # Quadratic objective
            cost = 0.5 * cs.norm_2(uncertified_action_var - u_var) ** 2

        # CBF constraint
        opti.subject_to(-self.linear_func(x=barrier_at_x)['y'] - lie_derivative_at_x - learned_residual <=
                        right_hand_side)

        # Input constraints
        for input_constraint in self.input_constraints_sym:
            opti.subject_to(input_constraint(u_var) <= 0)

        opti.minimize(cost)

        # Set verbosity option of optimizer
        opts = {'printLevel': 'low', 'error_on_fail': False}

        # Select QP solver
        opti.solver('qpoases', opts)

        self.opti_dict = {
            'opti': opti,
            'u_var': u_var,
            'slack_var': slack_var,
            'current_state_var': current_state_var,
            'uncertified_action_var': uncertified_action_var,
            'a_var': a_var,
            'b_var': b_var,
            'cost': cost
        }

    def solve_optimization(self,
                           current_state: np.ndarray,
                           uncertified_action: np.ndarray,
                           ) -> Tuple[np.ndarray, bool]:
        '''Solve the CBF optimization problem for a given observation and uncertified input.

        Args:
            current_state (np.ndarray): Current state/observation.
            uncertified_action (np.ndarray): The uncertified_controller's action.

        Returns:
            certified_action (np.ndarray): The certified action.
            feasible (bool): Whether the safety filtering was feasible or not.
        '''

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        u_var = opti_dict['u_var']
        slack_var = opti_dict['slack_var']
        current_state_var = opti_dict['current_state_var']
        uncertified_action_var = opti_dict['uncertified_action_var']
        a_var = opti_dict['a_var']
        b_var = opti_dict['b_var']
        cost = opti_dict['cost']

        opti.set_value(current_state_var, current_state)
        opti.set_value(uncertified_action_var, uncertified_action)

        a, b = self.extract_a_b(current_state)

        opti.set_value(a_var, a)
        opti.set_value(b_var, b)

        try:
            # Solve optimization problem
            sol = opti.solve()
            feasible = True
            certified_action = sol.value(u_var)
            if self.soft_constrained:
                slack_val = sol.value(slack_var)
                if slack_val > self.slack_tolerance:
                    print('Slack:', slack_val)
                    feasible = False
            cost = sol.value(cost)
        except RuntimeError as e:
            print(e)
            feasible = False
            certified_action = opti.debug.value(u_var)
            print('Certified_action:', certified_action)
            if self.soft_constrained:
                slack_val = opti.debug.value(slack_var)
                print('Slack:', slack_val)
            print('Lie Derivative:', self.lie_derivative(X=current_state, u=certified_action)['LfV'])
            print('Linear Function:', self.linear_func(x=self.cbf(X=current_state)['cbf'])['y'])
            print('------------------------------------------------')
        return certified_action, feasible

    def extract_a_b(self,
                    current_state: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Extracts the a and b vectors from the torch state.

        Args:
            current_state (np.ndarray): The current state of the system.

        Returns:
            a, b (np.ndarray): The a and b vectors used to calculate the learned residual.
        '''
        torch_state = torch.from_numpy(current_state)
        torch_state = torch.unsqueeze(torch_state, 0)
        torch_state = torch_state.to(torch.float32)
        a_b = self.mlp(torch_state)
        a_b = a_b.detach().numpy()
        a = a_b[0, :self.model.nu]
        b = a_b[0, -1]

        return a, b

    def compute_loss(self,
                     batch: dict
                     ) -> float:
        '''Compute training loss of the neural network that represents the Lie derivative error.

        Args:
            batch (dict): The batch of data used during training.

        Returns:
            loss (float): The loss of the batch.
        '''
        state, act, barrier_dot, barrier_dot_approx = batch['state'], batch['act'], batch['barrier_dot'], \
                                                      batch['barrier_dot_approx']

        # Predict a and b vectors
        a_b = self.mlp(state)
        a = torch.unsqueeze(a_b[:, 0], 1)
        b = torch.unsqueeze(a_b[:, 1], 1)

        # Determine the estimate of h_dot
        h_dot_estimate = barrier_dot + a * act + b

        # Determine loss
        loss = (h_dot_estimate - barrier_dot_approx).pow(2).mean()
        return loss

    def update(self,
               batch: dict
               ):
        '''Update the neural network parameters.

        Args:
            batch (dict): The batch of data used during training.
        '''
        loss = self.compute_loss(batch)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def reset(self):
        '''Resets the safety filter. '''
        super().reset()
        if hasattr(self, 'buffer'):
            self.buffer.reset()

    def save(self,
             path: str
             ):
        '''Saves model params and experiment state to path.

        Args:
            path (str): The path where to save the model params/experiment state.
        '''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)

        state_dict = {
            'agent': self.mlp.state_dict()
        }
        if self.training:
            exp_state = {
                'buffer': self.buffer.state_dict()
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self,
             path: str
             ):
        '''Restores model and experiment given path.

        Args:
            path (str): The path where the model params/experiment state are saved.
        '''
        state = torch.load(path)

        # Restore params
        self.mlp.load_state_dict(state['agent'])

        # Restore experiment state
        if self.training:
            self.buffer.load_state_dict(state['buffer'])

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Learn the error in the Lie derivative from multiple experiments.

        Args:
            env (BenchmarkEnv): The environment to be used for training.
        '''
        if env is None:
            env = self.env

        input_blending_weight = np.arange(self.num_episodes) / (self.num_episodes - 1)

        # Run experiments in loop
        for i in range(self.num_episodes):
            # Reset the episode
            obs, info = env.reset()

            counter = 0

            # Create arrays to collect data
            states = np.zeros((self.max_num_steps, self.model.nx))
            inputs = np.zeros((self.max_num_steps, self.model.nu))
            barrier_values = np.zeros((self.max_num_steps, 1))
            lie_derivative_values = np.zeros((self.max_num_steps, 1))
            lie_derivative_est = np.zeros((self.max_num_steps, 1))

            while counter < self.max_num_steps:
                print('Step: ', env.pyb_step_counter // self.step_size)

                # Determine safe action
                if self.uncertified_controller is None:
                    uncertified_action = self.env.action_space.sample()
                else:
                    uncertified_action = self.uncertified_controller.select_action(obs, info)
                safe_action, _ = self.certify_action(obs, uncertified_action)

                # Blend the safe and uncertified action
                blended_input = (1 - input_blending_weight[i]) * uncertified_action + input_blending_weight[i] * safe_action

                # Step the system
                obs, _, _, info = env.step(blended_input)
                print(f'obs: {obs}')
                print(f'action: {safe_action}')

                # Collect data
                states[counter, :] = obs
                inputs[counter, :] = blended_input
                barrier_values[counter, :] = self.cbf(X=obs)['cbf']
                lie_derivative_values[counter, :] = self.lie_derivative(X=obs, u=blended_input)['LfV']

                # Determine the estimated Lie derivative
                a, b = self.extract_a_b(current_state=obs)
                lie_derivative_est[counter, :] = lie_derivative_values[counter, :] + np.dot(a.T, blended_input) + b

                counter += 1

            print('Num time steps:', len(inputs))
            print('Certified control input weight:', input_blending_weight[i])

            # Numerical time differentiation (symmetric) of barrier function:
            barrier_dot_approx = (barrier_values[2:] - barrier_values[:-2]) / (2 * 1 / env.CTRL_FREQ)

            # Add data to buffer
            self.buffer.push({
                'state': states[1:-1, :],
                'act': inputs[1:-1, :],
                'barrier_dot': lie_derivative_values[1:-1, :],
                'barrier_dot_approx': barrier_dot_approx
            })

            # Update neural network parameters
            for _ in range(self.train_iterations):
                batch = self.buffer.sample(self.train_batch_size)
                self.update(batch)
