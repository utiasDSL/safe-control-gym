"""H2 Robust Control with Gaussian Process
Based on:
    *F. Berkenkamp and A. P. Schoellig, "Safe and robust learning control with Gaussian processes,"
     2015 European Control Conference (ECC), 2015, pp. 2496-2501, doi: 10.1109/ECC.2015.7330913.
"""


import scipy
import numpy as np
import casadi as cs
import cvxpy as cp
import pdb
import time
import torch
import gpytorch
import os

from skopt.sampler import Lhs
from sklearn.model_selection import train_test_split

from safe_control_gym.controllers.rc.gp_rc_utils import *
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system 
from safe_control_gym.controllers.mpc.gp_utils import GaussianProcessCollection, ZeroMeanIndependentGPModel
from safe_control_gym.envs.benchmark_env import Cost, Task


class GPRC(BaseController):
    """ Robust controller with Gaussian Process as dynamics residual. Only supports stabilization tasks currently.
    """
    def __init__(self,
                env_func,
                seed: int = 42,
                # Model args.
                q = [1],
                r = [1],
                custom_prior_dynamics : bool = False,
                learn_op: bool = False,
                # Training args.
                train_samples: int = 500,        
                validation_samples: int = 200,
                train_iterations: list = [1000],
                learning_rate: list = [0.1],
                use_gpu: bool = False,
                gp_model_path: str = None,
                # Optimization args.
                step_size: float = 0.1,
                max_optim_tries: int = 100,
                # Runner args.
                deque_size: int = 10,
                eval_batch_size: int = 1,
                # Shared/base args.
                output_dir="./results/temp",
                verbose=False,
                model_step_chk=False,
                random_init=True,
                save_data=False,
                data_dir=None,
                plot_traj=False,
                plot_dir=None,
                save_plot=False,
                init_state_randomization_info=None,
                **kwargs):
        """Initialize Gaussian Process H2 Robust Controller

        Args:
            env_func (gym.Env): Function to instantiate task/environment.
            q (list): Diagonals of state cost weight.
            r (list): Diagonals of input/action cost weight.
            custom_prior_dynamics (bool): Whether to use prior dynamics knowledge. If false, dynamics f(x, u) = x_{k} is used as prior.
            learn_op (bool): If True, operating point will be "unkown" and learned after training.
            train_samples (int): Number of training examples to use to train each GP.
            validation_samples (int): Number of points to use for the test set during training.
            train_iterations (list): Number of training iterations to train each dimension of the GP.  
            use_gpu (bool): Choose True to use GPU for training.
            gp_model_path (str): Path to pretrained GP.
            learning_rate (list): Learning rate for training each dimension of the GP.
            step_size (float): Step size to increase Beta parameter if LMI Optimization is infeasible.
            max_optim_tries (int): Maximum amount of attempts to obtain a solution for the LMI Optimization problem.
            deque_size (int): Number of episodes to average over per policy evaluation statistic.
            eval_batch_size (int): Number of episodes to run for policy evaluation.
            output_dir (str): Output directory to write logs and results.
            
        """
        # Algorithm specific args.
        for k, v in kwargs.items():
            self.__dict__[k] = v

        # Setup environment and environment related attributes:
        self.env_func = env_func
        self.env = env_func()
        self.env = RecordEpisodeStatistics(self.env, deque_size)
        self.task = self.env.TASK
        self.model = self.env.symbolic
        self.nx = self.model.nx #State dimension
        self.nu = self.model.nu #Input dimension
        self.dt = self.model.dt
        if not learn_op and self.task == Task.STABILIZATION:
            self.x_0 = self.env.X_GOAL
            self.u_0 = self.env.U_GOAL
        elif self.task == 'tracking':
            raise NotImplementedError("Tracking is not currently implemented.")
        else:
            #TODO: Set up operating point learning & update
            raise NotImplementedError("Learning the operating point is not currently implemented.")
        
        # Controller attributes:
        self.custom_prior_dynamics = custom_prior_dynamics
        self.Q_h2 = get_cost_weight_matrix(q, self.nx)
        self.R_h2 = get_cost_weight_matrix(r, self.nu)
        self.H2_gain = None
        
        # Setup initial controller to gather data:
        self.init_ctrl = InitCtrl(env_func, output_dir = output_dir,
                                plot_dir=plot_dir)

        # Training attributes:
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.train_iterations = train_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        try:
            self.gaussian_process = torch.load(os.path.join(self.gp_model_path, 'gp_model.pt'))
            print("Gaussian Process Loaded.")
        except:
            self.gaussian_process = None

        # Optimization attributes:
        self.step_size = step_size
        self.max_optim_tries = max_optim_tries

        # Runner attributes:
        self.deque_size = deque_size
        self.eval_batch_size = eval_batch_size
        self.logger = ExperimentLogger(output_dir)       
        self.random_init = random_init
        self.save_data = save_data
        self.data_dir = data_dir
        self.plot_traj = plot_traj
        self.plot_dir = plot_dir
        self.save_plot = save_plot
        self.init_state_randomization_info = init_state_randomization_info

        # Misc
        self.output_dir = output_dir
        self.seed = seed
        self.use_gpu = use_gpu
        self.model_step_chk = model_step_chk
        

    def init_prior_dynamics(self, use_custom_dynamics):
        """Sets up symbolic prior dynamics with actual control frequency.

        Args:
            use_custom_dynamics(bool): Whether to use a custom prior for dynamics.

        """     
        if not use_custom_dynamics:
            # Prior dynamics taken as f(x,u)=x
            dfdx = np.eye(self.nx)
            dfdu = np.zeros((self.nx, self.nu))
            delta_x = cs.MX.sym('delta_x', self.nx, 1)
            delta_u = cs.MX.sym('delta_u', self.nu, 1)
            x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
            self.prior_dynamics = cs.integrator(
                'linear_discrete_dynamics', self.model.integration_algo,
                {
                    'x': delta_x,
                    'p': delta_u,
                    'ode': x_dot_lin_vec
                }, {'tf': self.model.dt}
            )
            self.dfdx = dfdx
            self.dfdu = dfdu
            self.discrete_dfdx, self.discrete_dfdu = discretize_linear_system(dfdx, dfdu, self.dt, exact=False)

        else:
            #TODO: Implement other types of prior dynamics
            raise NotImplementedError("Custom prior dynamics not currently implemented.")

    def preprocess_training_data(self, x_seq, u_seq, x_next_seq):
        """Converts trajectory data for GP training.
        
        Args:
            x_seq (list): state sequence of np.array (nx,). 
            u_seq (list): action sequence of np.array (nu,). 
            x_next_seq (list): next state sequence of np.array (nx,). 
            
        Returns:
            inputs (np.array): inputs for GP training, (N, nx+nu).
            targets (np.array): targets for GP training, (N, nx).

        """
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.x_0 and self.u_0.
        x_pred_seq = self.prior_dynamics(x0=x_seq.T - self.x_0[:,np.newaxis],
                                               p=u_seq.T - self.u_0[:,np.newaxis])['xf'].toarray()
        
        targets = (x_next_seq.T - (x_pred_seq + self.x_0[:,np.newaxis])).transpose()  # (N, nx).
        inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).

        return inputs, targets

    def sample_init_states(self):
        """Use Latin Hypercube Sampling to generate initial state samples within environment bounds.

        Args:
            None

        Returns:
            init_state_samples(np.array): Initial state samples for init_ctrl to gather data.

        """ 
        lhs_sampler = Lhs(lhs_type='classic', criterion='maximin')
        limits = [(self.init_ctrl.env.INIT_STATE_RAND_INFO[key]['low'], self.init_ctrl.env.INIT_STATE_RAND_INFO[key]['high']) for key in
                 self.init_ctrl.env.INIT_STATE_RAND_INFO]
        samples = lhs_sampler.generate(limits, self.train_samples + self.validation_samples, random_state=self.seed)
        init_state_samples = np.array(samples)

        return init_state_samples

    def rollout_ctrl(self, ctrl=None, init_state_samples=None, samples=300):
        """Rolls out controller. May not give complete amount of samples if there are not enough init_states
        and rollouts are short.

        Args:
            ctrl (gym.Controller): Controller to use for rollout. If None, uses init_ctrl.
            init_state_samples (np.array): Initial states for each rollout.
            total_samples (int): Amount of samples needed.

        Returns:
            inputs (np.array): Inputs for learning model.
            targets (np.array): Targets for learning model.

        """
        if ctrl is None:
            ctrl = self.init_ctrl

        if len(init_state_samples.shape)==1:
            init_state_samples = init_state_samples[np.newaxis, :]

        inputs = np.empty((0, self.nx + self.nu))
        targets = np.empty((0, self.nx))

        i = 0
        """Rollouts may differ in length, so keep rolling out until total sample number is reached/surpassed
        or there are no more initial states"""
        while len(inputs) < samples and i < len(init_state_samples):        
            init_state = init_state_samples[i,:]
            run_env = self.env_func(init_state=init_state, randomized_init=False)
            eval_results = ctrl.run(run_env)
            x_obs = eval_results["obs"]
            u_seq = eval_results["action"][:-1,:]
            x_seq = x_obs[:-1,:]
            x_next_seq = x_obs[1:,:]
            run_env.close()
            input_batch, target_batch = self.preprocess_training_data(x_seq, u_seq, x_next_seq)
            inputs = np.append(inputs, input_batch, axis=0)
            targets = np.append(targets, target_batch, axis=0)
            print(f"Rollout: {i} | Data gathered so far: {len(inputs)}")
            i+=1    

        print(f"Rollouts done! | Amount data gathered: {len(inputs)}/{samples}")

        return inputs, targets

    def gather_training_data(self):
        """Rollout init_ctrl to gather states and inputs for GP training.

        Args:
            None

        Returns:
            train_inputs_tensor (torch.Tensor)
            train_targets_tensor (torch.Tensor)
            test_inputs_tensor (torch.Tensor)
            test_targets_tensor (torch.Tensor)

        """
        init_state_samples = self.sample_init_states()
        total_samples = self.train_samples + self.validation_samples    

        inputs, targets = self.rollout_ctrl(init_state_samples = init_state_samples, samples = total_samples)

        self.data_inputs = inputs[:total_samples]
        self.data_targets = targets[:total_samples]

        train_idx, test_idx = train_test_split(                                
                                list(range(inputs.shape[0])),
                                test_size=self.validation_samples/(self.train_samples+self.validation_samples),
                                random_state=self.seed
                                )

        train_inputs = inputs[train_idx, :]
        train_targets = targets[train_idx, :]
        self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
        test_inputs = inputs[test_idx, :]
        test_targets = targets[test_idx, :]
        self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}

        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        return train_inputs_tensor, train_targets_tensor, test_inputs_tensor, test_targets_tensor

    def solve_H2_optim(self, x_0, u_0, beta=1.):
        """Setups LMI and solves H2 Optimization problem in eq (22).
        
        Args:
            beta (float): Hyperparameter beta representing amount of uncertainty injected into the system. 

        """
        #Prediction query
        query = torch.tensor(np.hstack((x_0[:,np.newaxis].T, u_0[:,np.newaxis].T)))

        # Number of states and inputs
        n = self.nx
        m = self.nu

        # Generate matrices for LMI optimization problem 
        Bw = np.sqrt(self.gaussian_process.predict(query)[1].detach().numpy()) #Query prediction variances
        Cz = np.vstack([scipy.linalg.sqrtm(self.Q_h2), np.zeros((m,n))])
        Dz = np.vstack([np.zeros((n,m)), scipy.linalg.sqrtm(self.R_h2)])
        Cq = np.vstack([np.vstack(np.eye(n)*self.Au[:,np.newaxis,:]), np.zeros((n*m,n))])
        Dq = np.vstack([np.zeros((n*n,m)), np.vstack(np.eye(m)*self.Bu[:,np.newaxis])])
        Bp = np.hstack([np.kron(np.eye(n),np.ones((1,n))), np.kron(np.eye(n),np.ones((1,m)))])

        # Dimensions (as given in the paper)
        q = n
        r = n + m
        f = n**2 + n*m

        # Defining optimization variables
        W = cp.Variable((n+m,n+m), symmetric = True)
        Q = cp.Variable((n,n), symmetric = True)
        R = cp.Variable((m,n))
        lam = cp.Variable((n**2+n*m,n**2+n*m), diag = True)
        gam = cp.Variable()

        # Defining parameters
        bet = cp.Parameter(pos = True, value = beta)

        # Constraints
        mat_1 = cp.bmat([[W, Cz@Q + Dz@R],
                        [cp.transpose(Cz@Q + Dz@R), Q]])

        mat_2 = cp.bmat([[Q, np.zeros((n,q)), np.zeros((n,f)), cp.transpose(self.A@Q + self.B@R), cp.transpose(Cq@Q + Dq@R)],
                        [np.zeros((q,n)), np.eye(q), np.zeros((q,f)), cp.transpose(Bw), np.zeros((q,f))],
                        [np.zeros((f,n)), np.zeros((f,q)), lam, cp.transpose(Bp@lam), np.zeros((f,f))],
                        [self.A@Q + self.B@R, Bw, Bp@lam, Q, np.zeros((n,f))],
                        [Cq@Q + Dq@R, np.zeros((f,q)), np.zeros((f,f)), np.zeros((f,n)), bet*lam]])

        constraints = [cp.trace(W) <= gam, mat_1 >> 0, mat_2 >> 0]

        # Problem
        prob = cp.Problem(cp.Minimize(gam), constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)
        
        if prob.status=='optimal':
            print(f"LMI optimization solved with beta = {beta:.2f}.")
            return R.value, Q.value
        else:
            raise RuntimeError("LMI optimization infeasible.")

    def compute_H2_gain(self, x_0, u_0):
        # Linearization.
        self.init_lin_matrices(x_0, u_0)
        beta = 1.
        gain = None

        for i in range(self.max_optim_tries):
            try:
                R, Q = self.solve_H2_optim(x_0, u_0, beta)                
                gain = R@np.linalg.inv(Q)
                break
            except:
                beta += self.step_size
                print(f"H2 optimization infeasible, trying again with beta = {beta:.2f}")
        
        if gain is None:
            raise RuntimeError("Couldn't solve optimization problem. Try raising max number of tries.")

        return gain

    def init_lin_matrices(self, x_0, u_0):
        query = torch.tensor(np.hstack((x_0[:,np.newaxis].T, u_0[:,np.newaxis].T)))
        self.gp_jacob, self.gp_jacob_uncert = get_derivative_posterior(self.gaussian_process, query)    
        self.A =  self.discrete_dfdx + self.gp_jacob[:,:-self.nu]
        self.B =  self.discrete_dfdu + self.gp_jacob[:,-self.nu:]
        self.Au = self.gp_jacob_uncert[:,:-self.nu]
        self.Bu = self.gp_jacob_uncert[:,-self.nu:]

    def select_action(self, x):
        """Calculates control input u_k = K(x_k-x_s)+u_s.

        Args:
            x (np.array): step-wise observation/input.

        Returns:
           np.array: step-wise control input/action.

        """
        if self.task == Task.STABILIZATION:
            return self.H2_gain @ (x - self.x_0) + self.u_0
        elif self.task == Task.TRAJ_TRACKING:
            raise NotImplementedError("Tracking is not currently implemented.")
            self.H2_gain = compute_H2_gain(self.x_0[self.k], self.u_0)
            return self.H2_gain @ (x - self.x_0[self.k]) + self.u_0
        else:
            print(colored("Incorrect task specified.", "red"))

    def learn(self,
              input_data=None,
              target_data=None,
              gp_model_path=None,
              plot=False,
              save=False
              ):
        """Performs GP training.

        Args:
            input_data, target_data (optional, np.array): Data to use for training.
            gp_model_path (str): If not None, this is the path to pretrained models to use instead of training new ones.
            plot (bool): To plot validation trajectories or not.
            save (bool): Save gaussian process model.

        Returns:
            training_results (dict): Dictionary of the training results.

        """
        self.init_prior_dynamics(self.custom_prior_dynamics)

        train_inputs_tensor, train_targets_tensor, test_inputs_tensor, test_targets_tensor = self.gather_training_data()

        if gp_model_path is None:
            gp_model_path = self.gp_model_path

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
            ).double()

        #Use all GP dimensions
        target_dim = self.nx
        target_mask = list(range(self.nx))
        input_mask = list(range(self.nx + self.nu))
        self.gaussian_process = GaussianProcessCollection(model_type=ZeroMeanIndependentGPModel,
                                                     likelihood=likelihood,
                                                     target_dim=target_dim,
                                                     input_mask=input_mask,
                                                     target_mask=target_mask,
                                                     normalize=False
                                                     )

        if gp_model_path:
            self.gaussian_process.init_with_hyperparam(train_inputs_tensor,
                                                       train_targets_tensor,
                                                       gp_model_path)
        else:
            # Train the GP.
            if len(self.train_iterations)==1:
                train_iterations = self.train_iterations*target_dim
            else:
                train_iterations = self.train_iterations

            if len(self.learning_rate)==1:
                learning_rate = self.learning_rate*target_dim
            else:
                learning_rate = self.learning_rate

            self.gaussian_process.train(train_inputs_tensor,
                                        train_targets_tensor,
                                        test_inputs_tensor,
                                        test_targets_tensor,
                                        n_train=train_iterations,
                                        learning_rate=learning_rate,
                                        gpu=self.use_gpu,
                                        dir=self.output_dir)

        if plot:
            init_state = np.array([-1.0, 0.0, 0.0, 0.0])
            validation_inputs, validation_targets = self.rollout_ctrl(init_state_samples=init_state)
            fig_count = 0
            fig_count = self.gaussian_process.plot_trained_gp(torch.Tensor(validation_inputs).double(),
                                                              torch.Tensor(validation_targets).double(),
                                                              fig_count=fig_count)

        if save:
            torch.save(self.gaussian_process, os.path.join(self.output_dir,'gp_model.pt'))

        #Compute H2 gain after learning
        self.H2_gain = self.compute_H2_gain(self.x_0, self.u_0)

    def run(self, render=False, logging=False, verbose=False, use_adv=False):
        """Runs evaluation with current policy.

        Args:
            render (bool): if to render during the runs.
            logging (bool): if to log using logger during the runs.

        Returns:
            dict: evaluation results
            
        """
        ep_returns, ep_lengths = [], []
        frames = []
        self.ep_counter = 0
        self.k = 0

        # Reseed for batch-wise consistency.
        obs = self.env.reset()
        ep_seed = 1 #self.env.SEED

        while len(ep_returns) < self.eval_batch_size:
            # Current goal.
            if self.task == Task.STABILIZATION:
                current_goal = self.x_0
            elif self.task == Task.TRAJ_TRACKING:
                current_goal = self.x_0[self.k]

            # Select action.
            action = self.select_action(self.env.state)

            # Save initial condition.
            if self.k == 0:
                x_init = self.env.state
                if self.model_step_chk:
                    self.model_state = self.env.state

                # Initialize state and input stack.
                state_stack = self.env.state
                input_stack = action
                goal_stack = current_goal

                # Print initial state.
                print(colored("initial state (%d): " % ep_seed + get_arr_str(self.env.state), "green"))

            else:
                # Save state and input.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))

            # Step forward.
            obs, reward, done, info = self.env.step(action)

            # Debug with analytical model.
            if self.model_step_chk:
                self.model_step()

            # Update step counter
            self.k += 1

            if verbose:
                if self.task == Task.TRAJ_TRACKING:
                    print("goal state: " + get_arr_str(self.x_0))
                print("state: " + get_arr_str(self.env.state))
                if self.model_step_chk:
                    print("model_state: " + get_arr_str(self.model_state))
                print("obs: " + get_arr_str(obs))
                print("action: " + get_arr_str(action) + "\n")

            if render:
                self.env.render()
                frames.append(self.env.render("rgb_array"))

            if done:
                # Push last state and input to stack.
                # Note: the last input is not used.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))

                # Post analysis.
                if self.plot_traj or self.save_plot or self.save_data:
                    analysis_data = post_analysis(goal_stack, state_stack,
                                                  input_stack, self.env, 0,
                                                  self.ep_counter,
                                                  self.plot_traj,
                                                  self.save_plot,
                                                  self.save_data,
                                                  self.plot_dir, self.data_dir)
                    if self.ep_counter == 0:
                        ep_rmse = np.array([analysis_data["state_rmse_scalar"]])
                    else:
                        ep_rmse = np.vstack((ep_rmse, analysis_data["state_rmse_scalar"]))

                # Update iteration return and length lists.
                assert "episode" in info
                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])

                print(colored("Test Run %d reward %.2f" % (self.ep_counter, ep_returns[-1]), "yellow"))
                print(colored("initial state: " + get_arr_str(x_init), "yellow"))
                if self.task == Task.STABILIZATION:
                    print(colored("final state: " + get_arr_str(self.env.state),  "yellow"))
                    print(colored("goal state: " + get_arr_str(self.x_0), "yellow"))
                print(colored("==========================\n", "yellow"))

                # Save reward
                if self.save_data:
                    np.savetxt(self.data_dir + "test%d_rewards.csv" % self.ep_counter, np.array([ep_returns[-1]]), delimiter=',', fmt='%.8f')

                self.ep_counter += 1
                ep_seed += 1
                self.k = 0
                self.env = self.env_func(seed=ep_seed)
                self.env = RecordEpisodeStatistics(self.env, self.deque_size)
                obs = self.env.reset()

        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = "****** Evaluation ******\n"
            msg += "eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n".format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
            self.logger.info(msg + "\n")

        if self.save_data:
            np.savetxt(self.data_dir + "all_test_mean_rmse.csv", ep_rmse, delimiter=',', fmt='%.8f')

        eval_results = {"ep_returns": ep_returns, "ep_lengths": ep_lengths, "mse": np.square(ep_rmse), "obs":state_stack, "action":input_stack}
        if len(frames) > 0 and frames[0] is not None:
            eval_results["frames"] = frames
        return eval_results

    def reset(self):
        """Reset the controller before running.

        """
        # Dynamics model.
        if self.gaussian_process is not None:
            self.init_prior_dynamics(self.custom_prior_dynamics)
            self.H2_gain = self.compute_H2_gain(self.x_0, self.u_0)
        self.env.reset()
        self.init_ctrl.reset()

    def close(self):
        """Shuts down and cleans up lingering resources.
        
        """
        self.env.close()
        self.logger.close()