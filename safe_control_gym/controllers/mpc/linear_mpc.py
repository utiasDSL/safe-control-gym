"""Linear Model Predictive Control.

Based on:
    * https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/LQR.pdf
    * https://pythonrobotics.readthedocs.io/en/latest/modules/path_tracking.html#mpc-modeling 
    * https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py

"""
import numpy as np
import casadi as cs
import scipy.linalg
from sys import platform
from copy import deepcopy

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.controllers.cbf.cbf_qp_utils import cartesian_product
from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system, get_cost_weight_matrix


class LinearMPC(MPC):
    """ Simple linear MPC.
    
    """

    def __init__(
            self,
            env_func,
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            alpha=1,
            use_terminal_ingredients=False,
            warmstart=True,
            use_backup=False,
            output_dir="results/temp",
            additional_constraints=[],
            **kwargs):
        """Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): list of constraints.

        """
        # Store all params/args.
        for k, v in locals().items():
            if k != "self" and k != "kwargs" and "__" not in k:
                self.__dict__[k] = v

        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            **kwargs
        )

    def reset(self):
        """Prepares for training or evaluation.

        """
        self.X_LIN = np.atleast_2d(self.env.X_GOAL)[0,:].T
        self.U_LIN = np.atleast_2d(self.env.U_GOAL)[0,:]
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        self.compute_lqr_gain(self.X_LIN, self.U_LIN)
        super().reset()
    
    def compute_lqr_gain(self, x_0, u_0):
        # Linearization.
        df = self.model.df_func(x=x_0, u=u_0)
        A = df['dfdx'].toarray()
        B = df['dfdu'].toarray()
        # Compute controller gain.
        A, B = discretize_linear_system(A, B, self.model.dt)
        self.P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
        btp = np.dot(B.T, self.P)
        self.K = -1 * np.dot(np.linalg.inv(self.R + np.dot(btp, B)),
                           np.dot(btp, A))
        self.A = A
        self.B = B

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency.

        """
        # Original version, used in shooting.
        dfdxdfdu = self.model.df_func(x=self.X_LIN, u=self.U_LIN)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        delta_x = cs.MX.sym('delta_x', self.model.nx,1)
        delta_u = cs.MX.sym('delta_u', self.model.nu,1)
        x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        self.linear_dynamics_func = cs.integrator(
            'linear_discrete_dynamics', self.model.integration_algo,
            {
                'x': delta_x,
                'p': delta_u,
                'ode': x_dot_lin_vec
            }, {'tf': self.dt}
        )
        self.dfdx = dfdx
        self.dfdu = dfdu
    
    def is_valid_terminal_set(self, num_points=100):
        # Select the states to check the CBF condition
        max_bounds = np.array(self.constraints.state_constraints[0].upper_bounds)
        min_bounds = np.array(self.constraints.state_constraints[0].lower_bounds)

        # state dimension and input dimension
        nx, nu = self.model.nx, self.model.nu

        # Make sure that every vertex is checked
        num_points = max(2 * nx, num_points + num_points % (2 * nx))
        num_points_per_dim = num_points // nx

        # Create the lists of states to check
        states_to_sample = [np.linspace(min_bounds[i], max_bounds[i], num_points_per_dim) for i in range(nx)]
        states_to_check = cartesian_product(*states_to_sample)

        # Select the inputs to check
        max_bounds = np.array(self.constraints.input_constraints[0].upper_bounds)
        min_bounds = np.array(self.constraints.input_constraints[0].lower_bounds)

        # Make sure that every vertex is checked
        num_points = max(2 * nu, num_points + num_points % (2 * nu))
        num_points_per_dim = num_points // nu

        # Create the lists of inputs to check
        inputs_to_sample = [np.linspace(min_bounds[i], max_bounds[i], num_points_per_dim) for i in range(nu)]
        inputs_to_check = cartesian_product(*inputs_to_sample)

        invalid_states = []

        counter = 0
        volume_terminal_set = len(states_to_check)

        # Check if the set is valid for every considered state
        for state in states_to_check:
            counter += 1
            print(f'States Checked: {counter} / {len(states_to_check)}')

            terminal_cost = state.T @ self.P @ state
            if terminal_cost > self.alpha:
                volume_terminal_set -= 1
                continue

            for input in inputs_to_check:
                next_state = self.linear_dynamics_func(x0=state-self.X_LIN, p=input-self.U_LIN)['xf']
                next_terminal_cost = next_state.T @ self.P @ next_state
                stage_cost = state.T @ self.Q @ state + input.T @ self.R @ input
                success = next_terminal_cost - terminal_cost <= -stage_cost
                if success:
                    break

            if not success:
                invalid_states.append(state)

        print(f'Volume of terminal set: {float(volume_terminal_set) / len(states_to_check)}')

        valid_terminal_set = len(invalid_states) == 0
        return valid_terminal_set, invalid_states


    def setup_optimizer(self, unconstrained=False):
        """Sets up convex optimization problem.

        Including cost objective, variable bounds and dynamics constraints.

        """
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Cost (cumulative).
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            cost += cost_func(x=x_var[:, i]+self.X_LIN[:, None],
                              u=u_var[:, i]+self.U_LIN[:, None],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)["l"]
        # Terminal cost.
        if self.use_terminal_ingredients:
            cost += (x_var[:, -1]+self.X_LIN[:,None]-x_ref[:, -1]).T @ self.P @ (x_var[:, -1]+self.X_LIN[:,None]-x_ref[:, -1])
        else:
            cost += cost_func(x=x_var[:, -1]+self.X_LIN[:, None],
                              u=np.zeros((nu, 1))+self.U_LIN[:, None],
                              Xr=x_ref[:, -1],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)["l"]

        opti.minimize(cost)
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.linear_dynamics_func(x0=x_var[:, i], p=u_var[:,i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)
            # State and input constraints.
            if not unconstrained:
                for state_constraint in self.state_constraints_sym:
                    opti.subject_to(state_constraint(x_var[:,i] + self.X_LIN.T) < 0)
                for input_constraint in self.input_constraints_sym:
                    opti.subject_to(input_constraint(u_var[:,i] + self.U_LIN.T) < 0)
        # Final state constraints.
        if not unconstrained:
            for state_constraint in self.state_constraints_sym:
                opti.subject_to(state_constraint(x_var[:,-1] + self.X_LIN.T)  < 0)
            if self.use_terminal_ingredients:
                opti.subject_to((x_var[:, -1]+self.X_LIN[:,None]-x_ref[:, -1]).T @ self.P @ (x_var[:, -1]+self.X_LIN[:,None]-x_ref[:, -1]) <= self.alpha)
        # Initial condition constraints.
        opti.subject_to(x_var[:, 0] == x_init)
        # Create solver (IPOPT solver in this version).
        opts = {}
        if platform == "linux":
            opti.solver('sqpmethod', opts)
        elif platform == "darwin":
            opts = {"ipopt.max_iter": 100}
            opti.solver('ipopt', opts)
        else:
            print("[ERROR]: CasADi solver tested on Linux and OSX only.")
            exit()
        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "x_ref": x_ref,
            "cost": cost,
            "unconstrained": unconstrained
        }

    def select_action(self,
                      obs,
                      unconstrained=False
                      ):
        """Solve nonlinear mpc problem to get next action.
        
        Args:
            obs (np.array): current state/observation. 
            unconstrained (bool): whether to run the MPC without state and input constraints
        
        Returns:
            action (np.array): input/action to the task/env.

        """
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        opti_dict = self.opti_dict

        if opti_dict['unconstrained'] != unconstrained:
            self.setup_optimizer(unconstrained=unconstrained)
            opti_dict = self.opti_dict

        opti = opti_dict["opti"]
        x_var = opti_dict["x_var"]
        u_var = opti_dict["u_var"]
        x_init = opti_dict["x_init"]
        x_ref = opti_dict["x_ref"]
        cost = opti_dict["cost"]
        # Assign the initial state.
        opti.set_value(x_init, obs-self.X_LIN)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
        if self.warmstart and self.u_prev is not None and self.x_prev is not None:
            opti.set_initial(x_var, self.x_prev)
            opti.set_initial(u_var, self.u_prev)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
            self.x_prev = x_val
            self.u_prev = u_val
            self.results_dict['horizon_states'].append(deepcopy(self.x_prev) + self.X_LIN[:, None])
            self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev) + self.U_LIN[:, None])
        except RuntimeError as e:
            if self.use_backup and not unconstrained:
                action = self.select_action(obs, unconstrained=True)
                return action
            print(e)
            return_status = opti.return_status()
            if return_status == 'unknown':
                self.terminate_loop = True
                return None
            elif return_status == 'Maximum_Iterations_Exceeded':
                self.terminate_loop = True
                u_val = opti.debug.value(u_var)
        # Take first one from solved action sequence.
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        action += self.U_LIN
        self.prev_action = action

        return action
