"""Model Predictive Control

Example: 
    run mpc on cartpole balance::
    
        $ python tests/test_main.py --mode test_policy --exp_id mpc_cartpole \
        --algo mpc --task cartpole_200 --horizon 50 --q_mpc 100 0 10 0 
        
    run mpc on quadrotor hovering::
    
        $ python tests/test_main.py --mode test_policy --exp_id mpc_quad \
        --algo mpc --task quadrotor --ctrl_time_mult 10 --horizon 50 

Todo:
    * All for additional constraints to be added externally?
    * Remove eval batch?
    * Terminal Constraints? Talk to Lukas about how best to do this.
"""
import numpy as np
import casadi as cs
from copy import deepcopy

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import ConstraintList, GENERAL_CONSTRAINTS, create_ConstraintList_from_dict


# -----------------------------------------------------------------------------------
#                   (Nonlinear) MPC
# -----------------------------------------------------------------------------------


class MPC(BaseController):
    """MPC with full nonlinear model."""
    def __init__(
            self,
            env_func,
            # model args
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            warmstart=True,
            # runner args
            # shared/base args
            output_dir="results/temp",
            additional_constraints=None,
            **kwargs):
        """Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints

        """
        for k, v in locals().items():
            if k != "self" and k != "kwargs" and "__" not in k:
                self.__dict__.update({k: v})

        # task
        self.env = env_func()

        if additional_constraints is not None:
            additional_ConstraintsList = create_ConstraintList_from_dict(additional_constraints,
                                                                         GENERAL_CONSTRAINTS,
                                                                         self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.reset_constraints(self.env.constraints.constraints + self.additional_constraints)
        else:
            self.reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []

        # model params
        self.model = self.env.symbolic
        self.dt = self.model.dt

        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)

        # logging
        #self.logger = ExperimentLogger(output_dir)

    def reset_constraints(self, constraints):
        """ Setup the constraints list.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        self.constraints = ConstraintList(constraints)
        self.state_constraints_sym = self.constraints.get_state_constraint_symbolic_models()
        self.input_constraints_sym = self.constraints.get_input_constraint_symbolic_models()
        if len(self.constraints.input_state_constraints) > 0:
            raise NotImplementedError('MPC cannot handle combined state input constraints yet.')

    def add_constraints(self, constraints):
        """Add the constraints in provided list to the system

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        self.reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self, constraints):
        """Remove constraints from the current constraint list.

        Args:
            constraints (list): list of Constraints to be removed
        """
        old_constraints_list = self.constraints.constraints

        for constraint in constraints:
            assert constraint in self.constraints.constraints,\
                ValueError("This constraint is not in the current list of constraints")
            old_constraints_list.remove(constraint)
        self.reset_constraints(old_constraints_list)


    def close(self):
        """Cleans up resources."""
        self.env.close()
        #self.logger.close()

    def reset(self):
        """Prepares for training or evaluation."""
        # setup reference input
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            self.traj = self.env.X_GOAL.T
            # step along the reference
            self.traj_step = 0

        # dynamics model
        self.set_dynamics_func()

        # casadi optimizer
        self.setup_optimizer()
        # previously solved states & inputs, useful for warm start
        self.x_prev = None
        self.u_prev = None

        self.reset_results_dict()

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency."""
        self.dynamics_func = cs.integrator('fd', self.model.integration_algo, {
            'x': self.model.x_sym,
            'p': self.model.u_sym,
            'ode': self.model.x_dot
        }, {'tf': self.dt})

    def setup_optimizer(self):
        """Sets up nonlinear optimization problem."""
        nx, nu = self.model.nx, self.model.nu
        T = self.T

        # define optimizer and variables
        opti = cs.Opti()
        # states
        x_var = opti.variable(nx, T + 1)
        # inputs
        u_var = opti.variable(nu, T)
        # initial state
        x_init = opti.parameter(nx, 1)
        # reference (equilibrium point or trajectory, last step for terminal cost)
        x_ref = opti.parameter(nx, T + 1)

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # can ignore first state cost (since fist x_var == x_init)
            cost += cost_func(x=x_var[:, i],
                              u=u_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)["l"]
        # terminal cost
        cost += cost_func(x=x_var[:, -1],
                          u=np.zeros((nu, 1)),
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=self.Q,
                          R=self.R)["l"]
        opti.minimize(cost)

        # constraints
        for i in range(self.T):
            # dynamics constraints
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            for state_constraint in self.state_constraints_sym:
                opti.subject_to(state_constraint(x_var[:,i]) < 0)
            for input_constraint in self.input_constraints_sym:
                opti.subject_to(input_constraint(u_var[:,i]) < 0)

        # final state constraints
        # todo: Make this optional Terminal State Constraint
        for state_constraint in self.state_constraints_sym:
            opti.subject_to(state_constraint(x_var[:, i]) < 0)

        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        # create solver (IPOPT solver for now)
        # reference: https://groups.google.com/g/casadi-users/c/a9OGbF7R4ss
        opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}
        opti.solver('ipopt', opts)
        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "x_ref": x_ref,
            "cost": cost
        }

    def select_action(self, obs):
        """Solves nonlinear mpc problem to get next action.

        Args:
            obs (np.array): current state/observation. 
        
        Returns:
            np.array: input/action to the task/env. 

        """
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        x_var = opti_dict["x_var"]
        u_var = opti_dict["u_var"]
        x_init = opti_dict["x_init"]
        x_ref = opti_dict["x_ref"]
        cost = opti_dict["cost"]

        # assign the initial state
        opti.set_value(x_init, obs)

        # assign reference trajectory within horizon
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.mode == "tracking":
            self.traj_step += 1

        # initial guess for optim problem
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-1] = x_guess[:, 1:]
            u_guess[:-1] = u_guess[1:]

            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)

        # solve the optimization problem
        sol = opti.solve()
        x_val, u_val = sol.value(x_var), sol.value(u_var)
        self.x_prev = x_val
        self.u_prev = u_val

        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))

        # take the first one from solved action sequence
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        self.prev_action = action
        return action

    def get_references(self):
        """Constructs reference states along mpc horizon.(nx, T+1)"""
        if self.env.TASK == Task.STABILIZATION:
            # repeat goal state for horizon steps
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # slice trajectory for horizon steps,
            # if not long enough, repeat last state for remaining steps
            start = min(self.traj_step, self.traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))

            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception("Reference for this mode is not implemented.")
        return goal_states  # (nx, T+1)

    def reset_results_dict(self):
        self.results_dict = { 'obs': [],
                              'reward': [],
                              'done': [],
                              'info': [], # ignore initial info so we can save easier
                              'action': [],
                              'horizon_inputs': [],
                              'horizon_states': []
        }

    def run(self,
            env=None,
            render=False,
            logging=False,
            max_steps=100):
        """Runs evaluation with current policy.
        
        Args:
            render (bool): if to do real-time rendering. 
            logging (bool): if to log on terminal.
            
        Returns:
            dict: evaluation statisitcs, rendered frames. 

        """
        if env is None:
            env = self.env

        self.x_prev = None
        self.u_prev = None

        obs, info = env.reset()
        print("Init State:")
        print(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        self.reset_results_dict()
        self.results_dict['obs'].append(obs)

        i = 0
        if self.env.TASK == Task.STABILIZATION:
            MAX_STEPS = max_steps
        elif self.env.TASK == Task.TRAJ_TRACKING:
            MAX_STEPS = self.traj.shape[1]
        else:
            raise("Undefined Task")

        self.terminate_loop = False
        while np.linalg.norm(obs - env.X_GOAL) > 1e-3 and\
                i < MAX_STEPS and\
                not(self.terminate_loop):
            action = self.select_action(obs)
            if self.terminate_loop:
                print("Infeasible MPC Problem")
                break

            # repeat input for more efficient control
            obs, reward, done, info = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)

            print(i, '-th step.')
            print(action)
            print(obs)
            print(reward)
            print(done)
            print(info)
            print()

            if render:
                env.render()
                frames.append(env.render("rgb_array"))

            i += 1

        # collect evaluation results
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = "****** Evaluation ******\n"
            msg += "eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n".format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
            #self.logger.info(msg + "\n")

        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
        self.results_dict['action'] = np.vstack(self.results_dict['action'])

        return deepcopy(self.results_dict)
