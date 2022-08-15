'''Linear Quadratic Regulator (LQR)

[1] https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
[2] https://arxiv.org/pdf/1708.09342.pdf
'''

import numpy as np
from termcolor import colored

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import get_cost_weight_matrix, compute_lqr_gain, discretize_linear_system
from safe_control_gym.envs.benchmark_env import Task


class iLQR(BaseController):
    '''Linear quadratic regulator. '''

    def __init__(
            self,
            env_func,
            # Model args
            q_lqr: list = None,
            r_lqr: list = None,
            discrete_dynamics: bool = True,
            # iLQR args
            max_iterations: int = 15,
            lamb_factor: float = 10,
            lamb_max: float = 1000,
            epsilon: float = 0.01,
            **kwargs):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            q_lqr (list): diagonals of state cost weight.
            r_lqr (list): diagonals of input/action cost weight.
            discrete_dynamics (bool): if to use discrete or continuous dynamics.
            max_iterations, lamb_factor, lamb_max, epsilon: iLQR parameters.
        '''

        super().__init__(env_func, **kwargs)

        # Model parameters
        self.q_lqr = q_lqr
        self.r_lqr = r_lqr
        self.discrete_dynamics = discrete_dynamics

        # iLQR parameters
        self.max_iterations = max_iterations
        self.lamb_factor = lamb_factor
        self.lamb_max = lamb_max
        self.epsilon = epsilon

        self.env = env_func(info_in_reset=True)

        # Controller params.
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(self.q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)

        # Linearize at operating point (equilibrium for stabilization).
        self.x_0, self.u_0 = self.env.X_GOAL, self.env.U_GOAL

        if self.env.TASK == Task.STABILIZATION:
            self.gain = compute_lqr_gain(self.model, self.x_0, self.u_0,
                                         self.Q, self.R, self.discrete_dynamics)

        # Control stepsize.
        self.stepsize = self.model.dt

        self.ite_counter = 0
        self.input_ff_best = None
        self.gains_fb_best = None

        self.reset()

    def close(self):
        '''Cleans up resources. '''
        self.env.close()

    def learn(self, env=None, **kwargs):
        '''Run iLQR to iteratively update policy for each time step

        Args:
            env (gym.Env): the environment to be used for training
        '''

        if env is None:
            env = self.env

        # Initialize step size
        self.lamb = 1.0

        # Set update unstable flag to False
        self.update_unstable = False

        # Initialize previous cost
        self.previous_total_cost = -float('inf')

        # Loop through iLQR iterations
        while self.ite_counter < self.max_iterations:
            self.run(env=env, training=True)

            # Save data and update policy if iteration is finished.
            self.state_stack = np.vstack((self.state_stack, self.final_obs))

            print(colored(f'Iteration: {self.ite_counter}, Cost: {self.total_cost}', 'green'))
            print(colored('--------------------------', 'green'))

            if self.ite_counter == 0 and self.final_info['out_of_bounds']:
                print(colored('[ERROR] The initial policy might be unstable. '
                                + 'Break from iLQR updates.', 'red'))
                break

            # Maximum episode length.
            self.num_steps = np.shape(self.input_stack)[0]

            # Check if cost is increased and update lambda correspondingly
            delta_cost = self.total_cost - self.previous_total_cost
            if self.ite_counter == 0:
                # Save best iteration.
                self.best_iteration = self.ite_counter
                self.previous_total_cost = self.total_cost
                self.input_ff_best = np.copy(self.input_ff)
                self.gains_fb_best = np.copy(self.gains_fb)

                # Update controller gains
                self.update_policy(env)

                # Initialize improved flag.
                self.prev_ite_improved = False
            elif delta_cost > 0.0 or self.update_unstable:
                # If cost is increased, increase lambda
                self.lamb *= self.lamb_factor

                print(f'Cost increased by {-delta_cost}. '
                      + 'Set feedforward term and controller gain to that '
                      'from the previous iteration. '
                      f'Increased lambda to {self.lamb}.')
                print(f'Current policy is from iteration {self.best_iteration}.')

                # Reset feedforward term and controller gain to that from
                # the previous iteration.
                self.input_ff = np.copy(self.input_ff_best)
                self.gains_fb = np.copy(self.gains_fb_best)

                # Set improved flag to False.
                self.prev_ite_improved = False

                # Break if maximum lambda is reached.
                if self.lamb > self.lamb_max:
                    print(colored('Maximum lambda reached.', 'red'))
                    self.lamb = self.lamb_max

                # Reset update_unstable flag to False.
                self.update_unstable = False
            elif delta_cost <= 0.0:
                # Save feedforward term and gain and state and input stacks.
                self.best_iteration = self.ite_counter
                self.previous_total_cost = self.total_cost
                self.input_ff_best = np.copy(self.input_ff)
                self.gains_fb_best = np.copy(self.gains_fb)

                # Check consecutive cost increment (cost decrement).
                if abs(delta_cost) < self.epsilon and self.prev_ite_improved:
                    # Cost converged.
                    print(colored('iLQR cost converged with a tolerance '
                                  + f'of {self.epsilon}.', 'yellow'))
                    break

                # Set improved flag to True.
                self.prev_ite_improved = True

                # Update controller gains
                self.update_policy(env)

            self.ite_counter += 1

        print(colored(f'Best iteration: {self.best_iteration}', 'green'))

    def update_policy(self, env):
        '''Updates policy.

        Args:
            env (gym.Env): the environment to be used for training
        '''

        # Get symbolic loss function which also contains the necessary Jacobian
        # and Hessian of the loss w.r.t. state and input.
        loss = self.model.loss

        # Initialize backward pass.
        state_k = self.state_stack[-1]
        input_k = env.U_GOAL

        if env.TASK == Task.STABILIZATION:
            x_goal = self.x_0
        elif env.TASK == Task.TRAJ_TRACKING:
            x_goal = self.x_0[-1]
        loss_k = loss(x=state_k,
                      u=input_k,
                      Xr=x_goal,
                      Ur=env.U_GOAL,
                      Q=self.Q,
                      R=self.R)
        s = loss_k['l'].toarray()
        Sv = loss_k['l_x'].toarray().transpose()
        Sm = loss_k['l_xx'].toarray().transpose()

        # Backward pass.
        for k in reversed(range(self.num_steps)):
            # Get current operating point.
            state_k = self.state_stack[k]
            input_k = self.input_stack[k]

            # Linearized dynamics about (x_k, u_k).
            df_k = self.model.df_func(state_k, input_k)
            Ac_k, Bc_k = df_k[0].toarray(), df_k[1].toarray()
            Ad_k, Bd_k = discretize_linear_system(Ac_k, Bc_k, self.model.dt)

            # Get symbolic loss function that includes the necessary Jacobian
            # and Hessian of the loss w.r.t. state and input.
            if env.TASK == Task.STABILIZATION:
                x_goal = self.x_0
            elif env.TASK == Task.TRAJ_TRACKING:
                x_goal = self.x_0[k]
            loss_k = loss(x=state_k,
                          u=input_k,
                          Xr=x_goal,
                          Ur=env.U_GOAL,
                          Q=self.Q,
                          R=self.R)

            # Quadratic approximation of cost.
            q = loss_k['l'].toarray()  # l
            Qv = loss_k['l_x'].toarray().transpose()  # dl/dx
            Qm = loss_k['l_xx'].toarray().transpose()  # ddl/dxdx
            Rv = loss_k['l_u'].toarray().transpose()  # dl/du
            Rm = loss_k['l_uu'].toarray().transpose()  # ddl/dudu
            Pm = loss_k['l_xu'].toarray().transpose()  # ddl/dudx

            # Control dependent terms of cost function.
            g = Rv + Bd_k.transpose().dot(Sv)
            G = Pm + Bd_k.transpose().dot(Sm.dot(Ad_k))
            H = Rm + Bd_k.transpose().dot(Sm.dot(Bd_k))

            # Trick to make sure H is well-conditioned for inversion
            if not (np.isinf(np.sum(H)) or np.isnan(np.sum(H))):
                H = (H + H.transpose()) / 2
                H_eval, H_evec = np.linalg.eig(H)
                H_eval[H_eval < 0] = 0.0
                H_eval += self.lamb
                H_inv = np.dot(H_evec, np.dot(np.diag(1.0 / H_eval), H_evec.T))

                # Update controller gains.
                duff = -H_inv.dot(g)
                K = -H_inv.dot(G)

                # Update control input.
                input_ff_k = input_k + duff[:, 0] - K.dot(state_k)
                self.input_ff[:, k] = input_ff_k
                self.gains_fb[k] = K

                # Update s variables for time step k.
                Sm = Qm + Ad_k.transpose().dot(Sm.dot(Ad_k)) + \
                     K.transpose().dot(H.dot(K)) + \
                     K.transpose().dot(G) + G.transpose().dot(K)
                Sv = Qv + Ad_k.transpose().dot(Sv) + \
                     K.transpose().dot(H.dot(duff)) + K.transpose().dot(g) + \
                     G.transpose().dot(duff)
                s = q + s + 0.5 * duff.transpose().dot(H.dot(duff)) + \
                    duff.transpose().dot(g)
            else:
                self.update_unstable = True

    def select_action(self, obs, info=None, training=False):
        '''Determine the action to take at the current timestep.

        Args:
            obs (np.array): the observation at this timestep.
            info (list): the info at this timestep.

        Returns:
            action (np.array): the action chosen by the controller.
        '''

        step = self.extract_step(info)

        if training:
            if self.ite_counter == 0:
                action, gains_fb, input_ff = self.calculate_lqr_action(obs, step)
                # Save gains and feedforward term
                if step == 0:
                    self.gains_fb = gains_fb.reshape((1, self.model.nu, self.model.nx))
                    self.input_ff = input_ff.reshape(self.model.nu, 1)
                else:
                    self.gains_fb = np.append(self.gains_fb, gains_fb.reshape((1, self.model.nu, self.model.nx)), axis=0)
                    self.input_ff = np.append(self.input_ff, input_ff.reshape(self.model.nu, 1), axis=1)
            else:
                action = self.gains_fb[step].dot(obs) + self.input_ff[:, step]
        elif self.gains_fb_best is not None:
            action = self.gains_fb_best[step].dot(obs) + self.input_ff_best[:, step]
        else:
            action, _, _ = self.calculate_lqr_action(obs, step)

        return action

    def calculate_lqr_action(self, obs, step):
        '''Compute gain for the first iteration.
           action = -self.gain @ (x - self.x_0) + self.u_0

        Args:
            obs (ndarray): the observation at this timestep.
            step (int): the timestep.

        Returns:
            action (ndarray): The calculated action.
            gains_fb (ndarray): the feedback gains.
            input_ff (ndarray): the feedforward term.
        '''
        if self.env.TASK == Task.STABILIZATION:
            gains_fb = -self.gain
            input_ff = self.gain @ self.x_0 + self.u_0
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.gain = compute_lqr_gain(self.model, self.x_0[step],
                                        self.u_0, self.Q, self.R,
                                        self.discrete_dynamics)
            gains_fb = -self.gain
            input_ff = self.gain @ self.x_0[step] + self.u_0

        # Compute action
        action = gains_fb.dot(obs) + input_ff

        return action, gains_fb, input_ff

    def reset(self):
        '''Prepares for evaluation. '''
        self.env.reset()
        self.ite_counter = 0

    def run(self, env=None, max_steps=500, training=True):
        '''Runs evaluation with current policy.

        Args:
            env (gym.Env): environment for the task.
            max_steps (int): maximum number of steps
            training (bool): whether the algorithm is training or evaluating
        '''

        if env is None:
            env = self.env

        # Reseed for batch-wise consistency.
        obs, info = env.reset()
        total_cost = 0.0

        for step in range(max_steps):
            # Select action.
            action = self.select_action(obs=obs, info=info, training=training)

            # Save rollout data.
            if step == 0:
                # Initialize state and input stack.
                self.state_stack = obs
                self.input_stack = action
            else:
                # Save state and input.
                self.state_stack = np.vstack((self.state_stack, obs))
                self.input_stack = np.vstack((self.input_stack, action))

            # Step forward.
            obs, cost, done, info = env.step(action)
            total_cost -= cost

            if done:
                print(f'SUCCESS: Reached goal on step {step}. Terminating...')
                break

        self.final_obs = obs
        self.final_info = info
        self.total_cost = total_cost
