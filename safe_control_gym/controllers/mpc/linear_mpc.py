"""Linear Model Predictive Control.

Based on:
    * https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/LQR.pdf
    * https://pythonrobotics.readthedocs.io/en/latest/modules/path_tracking.html#mpc-modeling 
    * https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py

"""
import numpy as np
import casadi as cs

from sys import platform
from copy import deepcopy

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.envs.benchmark_env import Task


class LinearMPC(MPC):
    """ Simple linear MPC.
    
    """

    def __init__(
            self,
            env_func,
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            warmstart=True,
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
        self.X_LIN = np.atleast_2d(self.env.X_GOAL)[0,:].T
        self.U_LIN = np.atleast_2d(self.env.U_GOAL)[0,:]

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

    def setup_optimizer(self):
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
        cost += cost_func(x=x_var[:, -1]+self.X_LIN[:,None],
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
            for state_constraint in self.state_constraints_sym:
                opti.subject_to(state_constraint(x_var[:,i] + self.X_LIN.T) < 0)
            for input_constraint in self.input_constraints_sym:
                opti.subject_to(input_constraint(u_var[:,i] + self.U_LIN.T) < 0)
        # Final state constraints.
        for state_constraint in self.state_constraints_sym:
            opti.subject_to(state_constraint(x_var[:,-1] + self.X_LIN.T)  < 0)
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
            "cost": cost
        }

    def select_action(self,
                      obs
                      ):
        """Solve nonlinear mpc problem to get next action.
        
        Args:
            obs (np.array): current state/observation. 
        
        Returns:
            action (np.array): input/action to the task/env.

        """
        nx, nu = self.model.nx, self.model.nu
        T = self.T
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
