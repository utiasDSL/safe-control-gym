""" Robust (Tube) Model Predictive Control

Based on:
    * D. Mayne, M. Seron and S. Raković "Robust model predictive control of constrained linear
systems with bounded disturbance," in Automatica 41(2): 219–224. 2005. doi: https://doi.org/10.1016/
j.automatica.2004.08.019

"""

import numpy as np
import casadi as cs
import scipy.linalg
from pytope import Polytope

from sys import platform
from copy import deepcopy

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system, get_cost_weight_matrix
from safe_control_gym.controllers.mpc.mpc_utils import compute_min_RPI
from safe_control_gym.envs.constraints import LinearConstraint
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType

# Notes:
# state constraints must be polytopes, so far only tested BoundedConstraint

class TubeMPC(MPC):
    """Robust linear tube MPC 

    """

    def __init__(
            self,
            env_func,
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            wmax=[0.1],
            warmstart=True,
            output_dir="results/temp",
            additional_constraints=[],
            n_samples=600,
            sigma_confidence=3,
            eps_rpi=1e-5,
            s_max_rpi=50,
            mRPI=[0.1],
            compute_mRPI=True,
            vol_converge=1e-4,
            debug_mRPI=False,
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
            n_samples (int): number of samples used to learn disturbances.
            sigma_confidence (int): num of std devs to compute learned dist bound.
            eps_rpi (float): epsilon convergence parameter for computing mRPI (Mayne et al. 2005)
            s_max_rpi (int): max number of Minkowski additions for computing mRPI (Mayne et al. 2005)
            mRPI (list): users have the option of manually specifying the mRPI as the max point of a hypercube centered on the origin.
            compute_mRPI (bool): if True, mRPI ignored and we will calculate it, if False we will use the user-defined mRPI.
            vol_converge (float): mRPI algo converges when (vol - volprev) / vol < vol_converge
            debug_mRPI (bool): shows a plot of RPI volume vs. iterations and prints out information from algo

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
        if compute_min_RPI:
            if (self.env.NAME != 'quadrotor' and self.env.NAME != 'cartpole'):
                raise NotImplementedError("our current method to compute mRPI does not work for the chosen environment")
            if self.env.NAME == 'quadrotor':
                if self.env.QUAD_TYPE != QuadType.ONE_D:
                    raise NotImplementedError("our current method to compute mRPI does not work for the chosen environment")
        self.wmax = np.array(self.wmax)
        self.X_LIN = np.atleast_2d(self.env.X_GOAL)[0,:].T
        self.U_LIN = np.atleast_2d(self.env.U_GOAL)[0,:]
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        self.compute_lqr_gain(self.X_LIN, self.U_LIN)
        Ak = self.A + self.B @ self.K
        # Compute minimal Robust Positively Invariant (mRPI) set:
        if self.compute_mRPI:
            self.Z = compute_min_RPI(Ak, self.wmax, self.vol_converge, self.s_max_rpi, self.debug_mRPI)
        else:
            self.Z = Polytope(lb=-np.array(self.mRPI), ub=np.array(self.mRPI))
        super().reset()
        self.reset_constraint_polytopes()

    def reset_constraint_polytopes(self):
        temp_state_constraints = []
        for constraint in self.constraints.state_constraints:
            S = Polytope(lb=np.array(constraint.lower_bounds),
                         ub=np.array(constraint.upper_bounds))
            S = S - self.Z
            temp_state_constraints.append(LinearConstraint(self.env, S.A, S.b, 'state'))
        self.constraints.state_constraints = temp_state_constraints

        temp_input_constraints = []
        for constraint in self.constraints.input_constraints:
            U = Polytope(lb=np.array(constraint.lower_bounds),
                         ub=np.array(constraint.upper_bounds))
            U = U - self.K * self.Z
            temp_input_constraints.append(LinearConstraint(self.env, U.A, U.b, 'input'))
        self.constraints.input_constraints = temp_input_constraints
        self.reset_constraints(self.constraints.state_constraints + self.constraints.input_constraints)

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

    def setup_optimizer(self, x_init=None):
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
        if x_init is None:
            x_init = np.zeros((self.A.shape[0], 1))
        X_init = x_init + self.Z  # Minkowski addition with polytope Z.
        init_con = LinearConstraint(self.env, X_init.A, X_init.b, 'state')
        init_symb_con = init_con.get_symbolic_model()

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
        # Initial state constraints
        opti.subject_to(init_symb_con(x_var[:, 0]) < 0)
        # Final state constraints.
        for state_constraint in self.state_constraints_sym:
            opti.subject_to(state_constraint(x_var[:,-1] + self.X_LIN.T)  < 0)
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
        self.setup_optimizer(x_init=obs-self.X_LIN)
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        x_var = opti_dict["x_var"]
        u_var = opti_dict["u_var"]
        x_ref = opti_dict["x_ref"]

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
        action += self.K @ (obs - self.X_LIN - x_val[:, 0])
        self.prev_action = action
        return action

    def learn(self,
              env=None
              ):
        """Compute the bounded disturbance set.
        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        """

        if env is None:
            env = self.training_env

        # Create set of error residuals.
        w = np.zeros((self.model.nx, self.n_samples))

        # Use uniform sampling of control inputs and states.
        for i in range(self.n_samples):
            init_state, _ = env.reset()
            if self.env.NAME == 'quadrotor':
                u = np.random.rand(self.model.nu)/8 - 1/16 + self.U_LIN
            else:
                u = env.action_space.sample() # Will yield a random action within action space.
            x_next_obs, _, _, _ = env.step(u)
            x_next_estimated = self.linear_dynamics_func(x0=init_state, p=u)['xf'].toarray()
            w[:,i] = x_next_obs - x_next_estimated[:,0]

        self.wmax = np.mean(w.T, axis=0) + self.sigma_confidence*np.std(w.T, axis=0)
        self.Z = Polytope(lb=-self.wmax, ub=self.wmax)
        self.reset_constraint_polytopes()
        print('Learned disturbance set')
