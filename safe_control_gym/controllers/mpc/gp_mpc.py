"""Model Predictive Control with a Gaussian Process model.

Based on:
    * L. Hewing, J. Kabzan and M. N. Zeilinger, "Cautious Model Predictive Control Using Gaussian Process Regression,"
     in IEEE Transactions on Control Systems Technology, vol. 28, no. 6, pp. 2736-2743, Nov. 2020, doi: 10.1109/TCST.2019.2949757.

Implementation details:
    1. The previous time step MPC solution is used to compute the set constraints and GP dynamics rollout.
       Here, the dynamics are rolled out using the Mean Equivelence method, the fastest, but least accurate.
    2. The GP is approximated using the Fully Independent Training Conditional (FITC) outlined in
        * J. Quinonero-Candela, C. E. Rasmussen, and R. Herbrich, “A unifying view of sparse approximate Gaussian process regression,”
          Journal of Machine Learning Research, vol. 6, pp. 1935–1959, 2005.
          https://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
        * E. Snelson and Z. Ghahramani, “Sparse gaussian processes using pseudo-inputs,” in Advances in Neural Information Processing
          Systems, Y. Weiss, B. Scholkopf, and J. C. Platt, Eds., 2006, pp. 1257–1264.
       and the inducing points are the previous MPC solution.
    3. Each dimension of the learned error dynamics is an independent Zero Mean SE Kernel GP.

"""
import munch
import scipy
import numpy as np
import casadi as cs
from copy import deepcopy
import time
import torch
import gpytorch
from skopt.sampler import Lhs
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min

from safe_control_gym.controllers.mpc.linear_mpc import LinearMPC, MPC
from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import GaussianProcessCollection, ZeroMeanIndependentGPModel, covSEard, kmeans_centriods
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS

class GPMPC(MPC):
    """MPC with Gaussian Process as dynamics residual. 

    """

    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            additional_constraints: list = None,
            soft_constraints: dict = None,
            warmstart: bool = True,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            sparse_gp: bool = False,
            n_ind_points: int = 150,
            inducing_point_selection_method: str = 'kmeans',
            recalc_inducing_points_at_every_step: bool = False,
            online_learning: bool = False,
            inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = "results/temp",
            **kwargs
            ):
        """Initialize GP-MPC.

        Args:
            env_func (gym.Env): functionalized initialization of the environment.
            seed (int): random seed.
            horizon (int): MPC planning horizon.
            Q, R (np.array): cost weight matrix.
            use_prev_start (bool): Warmstart mpc with the previous solution.
            train_iterations (int): the number of training examples to use for each dimension of the GP.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            optimization_iterations (list): the number of optimization iterations for each dimension of the GP.
            learning_rate (list): the learning rate for training each dimension of the GP.
            normalize_training_data (bool): Normalize the training data.
            use_gpu (bool): use GPU while training the gp.
            gp_model_path (str): path to a pretrained GP model. If None, will train a new one.
            output_dir (str): directory to store model and results.
            prob (float): desired probabilistic safety level.
            initial_rollout_std (float): the initial std (across all states) for the mean_eq rollout.
            inertial_prop (list): to initialize the inertial properties of the prior model.
            prior_param_coeff (float): constant multiplying factor to adjust the prior model intertial properties.
            input_mask (list): list of which input dimensions to use in GP model. If None, all are used.
            target_mask (list): list of which output dimensions to use in the GP model. If None, all are used.
            gp_approx (str): 'mean_eq' used mean equivalence rollout for the GP dynamics. Only one that works currently.
            sparse_gp (bool): True to use sparse GP approximations, otherwise no spare approximation is used.
            n_ind_points (int): Number of inducing points to use got the FTIC gp approximation.
            inducing_point_selection_method (str): kmeans for kmeans clustering, 'random' for random.
            recalc_inducing_points_at_every_step (bool): True to recompute the gp approx at every time step.
            online_learning (bool): if true, GP kernel values will be updated using past trajectory values.
            additional_constraints (list): list of Constraint objects defining additional constraints to be used.

        """
        if type(inertial_prop) is list:
            self.prior_env_func = partial(env_func,
                                          inertial_prop=np.array(inertial_prop)*prior_param_coeff,
                                          prior_prop=np.array(inertial_prop)*prior_param_coeff)
        else:
            assert (type(inertial_prop) is dict) or (type(inertial_prop) == munch.Munch), '[Error]: GPMPC.__init__(): Intertial properties must be a list, dict, or munch.Munch'
            inertial_prop.update((prop, val*prior_param_coeff) for prop, val in inertial_prop.items())
            self.prior_env_func = partial(env_func,
                                          inertial_prop=inertial_prop,
                                          prior_prop=inertial_prop)
        if soft_constraints is None:
            self.soft_constraints_params = {'gp_soft_constraints': False,
                                            'gp_soft_constraints_coeff': 0,
                                            'prior_soft_constraints': False,
                                            'prior_soft_constraints_coeff': 0}
        else:
            self.soft_constraints_params = soft_constraints

        # Initialize the method using linear MPC.
        self.prior_ctrl = LinearMPC(
            self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
            terminate_run_on_done=terminate_run_on_done,
            # runner args
            # shared/base args
            output_dir=output_dir,
            additional_constraints=additional_constraints,
        )
        self.prior_ctrl.reset()
        super().__init__(
            self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=self.soft_constraints_params['gp_soft_constraints'],
            terminate_run_on_done=terminate_run_on_done,
            # runner args
            # shared/base args
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            **kwargs)
        # Setup environments.
        self.env_func = env_func
        self.env = env_func(randomized_init=False, seed=seed)
        self.env_training = env_func(randomized_init=True, seed=seed)
        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.data_inputs = None
        self.data_targets = None
        self.prior_dynamics_func = self.prior_ctrl.linear_dynamics_func
        self.X_EQ = np.atleast_2d(self.env.X_EQ)[0,:].T
        self.U_EQ = np.atleast_2d(self.env.U_EQ)[0,:].T
        # GP and training parameters.
        self.gaussian_process = None
        self.train_iterations = train_iterations
        self.test_data_ratio = test_data_ratio
        self.overwrite_saved_data = overwrite_saved_data
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        self.normalize_training_data = normalize_training_data
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.seed = seed
        self.prob = prob
        self.sparse_gp = sparse_gp
        if input_mask is None:
            self.input_mask = np.arange(self.model.nx + self.model.nu).tolist()
        else:
            self.input_mask = input_mask
        if target_mask is None:
            self.target_mask = np.arange(self.model.nx).tolist()
        else:
            self.target_mask = target_mask
        Bd = np.eye(self.model.nx)
        self.Bd = Bd[:, self.target_mask]
        self.gp_approx = gp_approx
        self.n_ind_points = n_ind_points
        assert inducing_point_selection_method in ['kmeans', 'random'], '[Error]: Inducing method choice is incorrect.'
        self.inducing_point_selection_method = inducing_point_selection_method
        self.recalc_inducing_points_at_every_step = recalc_inducing_points_at_every_step
        self.online_learning = online_learning
        self.last_obs = None
        self.last_action = None
        self.initial_rollout_std = initial_rollout_std
        # MPC params
        self.gp_soft_constraints = self.soft_constraints_params['gp_soft_constraints']
        self.gp_soft_constraints_coeff = self.soft_constraints_params['gp_soft_constraints_coeff']

    def setup_prior_dynamics(self):
        """Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics.

        """
        # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
        A, B = discretize_linear_system(self.prior_ctrl.dfdx, self.prior_ctrl.dfdu, self.dt)
        Q_lqr = self.Q
        R_lqr = self.R
        P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
        btp = np.dot(B.T, P)
        self.lqr_gain = -np.dot(np.linalg.inv(self.R + np.dot(btp, B)), np.dot(btp, A))
        self.discrete_dfdx = A
        self.discrete_dfdu = B

    def set_gp_dynamics_func(self, n_ind_points):
        """Updates symbolic dynamics.

        With actual control frequency, initialize GP model and add to the combined dynamics.

        """
        self.setup_prior_dynamics()
        # Compute the probabilistic constraint inverse CDF according to section III.D.b in Hewing 2019.
        self.inverse_cdf = scipy.stats.norm.ppf(1 - (1/self.model.nx - (self.prob + 1)/(2*self.model.nx)))
        self.create_sparse_GP_machinery(n_ind_points)

    def create_sparse_GP_machinery(self, n_ind_points):
        """This setups the gaussian process approximations for FITC formulation.

        """
        lengthscales, signal_var, noise_var, gp_K_plus_noise = self.gaussian_process.get_hyperparameters(as_numpy=True)
        self.length_scales = lengthscales.squeeze()
        self.signal_var = signal_var.squeeze()
        self.noise_var = noise_var.squeeze()
        self.gp_K_plus_noise = gp_K_plus_noise
        Nx = len(self.input_mask)
        Ny = len(self.target_mask)
        N = self.gaussian_process.n_training_samples
        # Create CasADI function for computing the kernel K_z_zind with parameters for z, z_ind, length scales and signal variance.
        # We need the CasADI version of this so that it can by symbolically differentiated in in the MPC optimization.
        z1 = cs.SX.sym('z1', Nx)
        z2 = cs.SX.sym('z2', Nx)
        ell_s = cs.SX.sym('ell', Nx)
        sf2_s = cs.SX.sym('sf2')
        z_ind  = cs.SX.sym('z_ind', n_ind_points, Nx)
        covSE = cs.Function('covSE', [z1, z2, ell_s, sf2_s],
                            [covSEard(z1, z2, ell_s, sf2_s)])
        ks = cs.SX.zeros(1, n_ind_points)
        for i in range(n_ind_points):
            ks[i] = covSE(z1, z_ind[i, :], ell_s, sf2_s)
        ks_func = cs.Function('K_s', [z1, z_ind, ell_s, sf2_s], [ks])
        K_z_zind = cs.SX.zeros(Ny, n_ind_points)
        for i in range(Ny):
            K_z_zind[i,:] = ks_func(z1, z_ind, self.length_scales[i,:], self.signal_var[i])
        # This will be mulitplied by the mean_post_factor computed at every time step to compute the approximate mean.
        self.K_z_zind_func = cs.Function('K_z_zind', [z1, z_ind],[K_z_zind],['z1', 'z2'],['K'])

    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_next_seq
                                 ):
        """Converts trajectory data for GP trianing.
        
        Args:
            x_seq (list): state sequence of np.array (nx,). 
            u_seq (list): action sequence of np.array (nu,). 
            x_next_seq (list): next state sequence of np.array (nx,). 
            
        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).

        """
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        x_pred_seq = self.prior_dynamics_func(x0=x_seq.T - self.prior_ctrl.X_EQ[:, None],
                                               p=u_seq.T - self.prior_ctrl.U_EQ[:,None])['xf'].toarray()
        targets = (x_next_seq.T - (x_pred_seq+self.prior_ctrl.X_EQ[:,None])).transpose()  # (N, nx).
        inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).
        return inputs, targets

    def precompute_probabilistic_limits(self,
                                        print_sets=False
                                        ):
        """This updates the constraint value limits to account for the uncertainty in the dynamics rollout.

        Args:
            print_sets (bool): True to print out the sets for debugging purposes.

        """
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        state_covariances = np.zeros((self.T+1, nx, nx))
        input_covariances = np.zeros((self.T, nu, nu))
        # Initilize lists for the tightening of each constraint.
        state_constraint_set = []
        for state_constraint in self.constraints.state_constraints:
            state_constraint_set.append(np.zeros((state_constraint.num_constraints, T+1)))
        input_constraint_set = []
        for input_constraint in self.constraints.input_constraints:
            input_constraint_set.append(np.zeros((input_constraint.num_constraints, T)))
        if self.x_prev is not None and self.u_prev is not None:
            #cov_x = np.zeros((nx, nx))
            cov_x = np.diag([self.initial_rollout_std**2]*nx)
            for i in range(T):
                state_covariances[i] = cov_x
                cov_u = self.lqr_gain @ cov_x @ self.lqr_gain.T
                input_covariances[i] = cov_u
                cov_xu = cov_x @ self.lqr_gain.T
                z = np.hstack((self.x_prev[:,i], self.u_prev[:,i]))
                if self.gp_approx == 'taylor':
                    raise NotImplementedError("Taylor GP approximation is currently not working.")
                elif self.gp_approx == 'mean_eq':
                    _, cov_d_tensor = self.gaussian_process.predict(z[None,:], return_pred=False)
                    cov_d = cov_d_tensor.detach().numpy()
                    # ToDo: Addition of noise here! And do we still need initial_rollout_std
                    _, _, cov_noise, _ = self.gaussian_process.get_hyperparameters()
                    cov_d = cov_d + np.diag(cov_noise.detach().numpy())
                else:
                    raise NotImplementedError('gp_approx method is incorrect or not implemented')
                # Loop through input constraints and tighten by the required ammount.
                for ui, input_constraint in enumerate(self.constraints.input_constraints):
                    input_constraint_set[ui][:, i] = -1*self.inverse_cdf * \
                                                    np.absolute(input_constraint.A) @ np.sqrt(np.diag(cov_u))
                for si, state_constraint in enumerate(self.constraints.state_constraints):
                    state_constraint_set[si][:, i] = -1*self.inverse_cdf * \
                                                    np.absolute(state_constraint.A) @ np.sqrt(np.diag(cov_x))
                if self.gp_approx == 'taylor':
                    raise NotImplementedError("Taylor GP rollout not implemented.")
                elif self.gp_approx == 'mean_eq':
                    # Compute the next step propogated state covariance using mean equivilence.
                    cov_x = self.discrete_dfdx @ cov_x @ self.discrete_dfdx.T + \
                            self.discrete_dfdx @ cov_xu @ self.discrete_dfdu.T + \
                            self.discrete_dfdu @ cov_xu.T @ self.discrete_dfdx.T + \
                            self.discrete_dfdu @ cov_u @ self.discrete_dfdu.T + \
                            self.Bd @ cov_d @ self.Bd.T
                else:
                    raise NotImplementedError('gp_approx method is incorrect or not implemented')
            # Udate Final covariance.
            for si, state_constraint in enumerate(self.constraints.state_constraints):
                state_constraint_set[si][:,-1] = -1 * self.inverse_cdf * \
                                                np.absolute(state_constraint.A) @ np.sqrt(np.diag(cov_x))
            state_covariances[-1] = cov_x
        if print_sets:
            print("Probabilistic State Constraint values along Horizon:")
            print(state_constraint_set)
            print("Probabilistic Input Constraint values along Horizon:")
            print(input_constraint_set)
        self.results_dict['input_constraint_set'].append(input_constraint_set)
        self.results_dict['state_constraint_set'].append(state_constraint_set)
        self.results_dict['state_horizon_cov'].append(state_covariances)
        self.results_dict['input_horizon_cov'].append(input_covariances)
        return state_constraint_set, input_constraint_set

    def precompute_mean_post_factor_all_data(self):
        """If the number of data points is less than the number of inducing points, use all the data
        as kernel points.
        """
        dim_gp_outputs = len(self.target_mask)
        n_training_samples = self.train_data['train_targets'].shape[0]
        inputs = self.train_data['train_inputs']
        targets = self.train_data['train_targets']
        mean_post_factor = np.zeros((dim_gp_outputs, n_training_samples))
        for i in range(dim_gp_outputs):
            K_z_z = self.gaussian_process.K_plus_noise_inv[i]
            mean_post_factor[i] = K_z_z.detach().numpy() @ targets[:,self.target_mask[i]]

        return mean_post_factor, inputs[:, self.input_mask]

    def precompute_sparse_gp_values(self, n_ind_points):
        """Uses the last MPC solution to precomupte values associated with the FITC GP approximation.

        Args:
            n_ind_points (int): Number of inducing points.
        """
        n_data_points = self.gaussian_process.n_training_samples
        dim_gp_inputs = len(self.input_mask)
        dim_gp_outputs = len(self.target_mask)
        inputs = self.train_data['train_inputs']
        targets = self.train_data['train_targets']
        # Get the inducing points.
        if False and self.x_prev is not None and self.u_prev is not None:
            # Use the previous MPC solution as in Hewing 2019.
            z_prev = np.hstack((self.x_prev[:,:-1].T, self.u_prev.T))
            z_prev = z_prev[:,self.input_mask]
            inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points-self.T, replace=False)
            #z_ind = self.data_inputs[inds][:, self.input_mask]
            z_ind = np.vstack((z_prev,inputs[inds][:, self.input_mask]))
        else:
            # If there is no previous solution. Choose T random training set points.
            if self.inducing_point_selection_method == 'kmeans':
                centroids = kmeans_centriods(n_ind_points, inputs[:, self.input_mask], rand_state=self.seed)
                inds, dist_mat = pairwise_distances_argmin_min(centroids, inputs[:, self.input_mask])
                z_ind = inputs[inds][:, self.input_mask]
            elif self.inducing_point_selection_method == 'random':
                inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points, replace=False)
                z_ind = inputs[inds][:, self.input_mask]
            else:
                raise ValueError("[Error]: gp_mpc.precompute_sparse_gp_values: Only 'kmeans' or 'random' allowed.")
        K_zind_zind = self.gaussian_process.kernel(torch.Tensor(z_ind).double())
        K_zind_zind_inv = self.gaussian_process.kernel_inv(torch.Tensor(z_ind).double())
        K_x_zind = self.gaussian_process.kernel(torch.from_numpy(inputs[:, self.input_mask]).double(),
                                                torch.tensor(z_ind).double())
        #Q_X_X = K_x_zind @ K_zind_zind_inv @ K_x_zind.transpose(1,2)
        Q_X_X = K_x_zind @ torch.linalg.solve(K_zind_zind, K_x_zind.transpose(1,2))
        Gamma = torch.diagonal(self.gaussian_process.K_plus_noise - Q_X_X, 0, 1, 2)
        Gamma_inv = torch.diag_embed(1/Gamma)
        # Todo: Should inverse be used here instead? pinverse was more stable previsouly.
        Sigma_inv = K_zind_zind + K_x_zind.transpose(1,2) @ Gamma_inv @ K_x_zind
        Sigma = torch.pinverse(K_zind_zind + K_x_zind.transpose(1,2) @ Gamma_inv @ K_x_zind)
        mean_post_factor = torch.zeros((dim_gp_outputs, n_ind_points))
        for i in range(dim_gp_outputs):
            mean_post_factor[i] = torch.linalg.solve(Sigma_inv[i], K_x_zind[i].T @ Gamma_inv[i] @
                                  torch.from_numpy(targets[:,self.target_mask[i]]).double())
            #mean_post_factor[i] = Sigma[i] @ K_x_zind[i].T @ Gamma_inv[i] @ torch.from_numpy(targets[:, self.target_mask[i]]).double()
        return mean_post_factor.detach().numpy(), Sigma_inv.detach().numpy(), K_zind_zind_inv.detach().numpy(), z_ind
        #return mean_post_factor.detach().numpy(), Sigma.detach().numpy(), K_zind_zind_inv.detach().numpy(), z_ind

    def setup_gp_optimizer(self, n_ind_points):
        """Sets up nonlinear optimization problem including cost objective, variable bounds and dynamics constraints.

        Args:
            n_ind_points (int): Number of inducing points.

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
        # Add slack variables
        if self.gp_soft_constraints:
            state_slack_list = []
            for state_constraint in self.constraints.state_constraints:
                state_slack_list.append(opti.variable(state_constraint.num_constraints, T+1))
            input_slack_list = []
            for input_constraint in self.constraints.input_constraints:
                input_slack_list.append(opti.variable(input_constraint.num_constraints, T))
            soft_con_coeff = self.gp_soft_constraints_coeff
        # Chance constraint limits.
        state_constraint_set = []
        for state_constraint in self.constraints.state_constraints:
            state_constraint_set.append(opti.parameter(state_constraint.num_constraints, T+1))
        input_constraint_set = []
        for input_constraint in self.constraints.input_constraints:
            input_constraint_set.append(opti.parameter(input_constraint.num_constraints, T))
        # Sparse GP mean postfactor matrix.
        mean_post_factor = opti.parameter(len(self.target_mask), n_ind_points)

        # Sparse GP inducing points.
        z_ind = opti.parameter(n_ind_points, len(self.input_mask))
        # Cost (cumulative).
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            cost += cost_func(x=x_var[:, i],
                              u=u_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)["l"]
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1],
                          u=np.zeros((nu, 1)),
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=self.Q,
                          R=self.R)["l"]
        z = cs.vertcat(x_var[:,:-1], u_var)
        z = z[self.input_mask,:]

        # Constraints
        for i in range(self.T):
            # Dynamics constraints using the dynamics of the prior and the mean of the GP.
            # This follows the tractable dynamics formulation in Section III.B in Hewing 2019.
            # Note that for the GP approximation, we are purposely using elementwise multiplication *.
            if True and self.sparse_gp:
                next_state = self.prior_dynamics_func(x0=x_var[:, i]-self.prior_ctrl.X_EQ[:,None],
                                                      p=u_var[:, i]-self.prior_ctrl.U_EQ[:,None])['xf'] + \
                self.prior_ctrl.X_EQ[:,None]+ self.Bd @ cs.sum2(self.K_z_zind_func(z1=z[:,i].T, z2=z_ind)['K'] * mean_post_factor)
            else:
                # Sparse GP approximation doesn't always work well, thus, use Exact GP regression. This is much slower,
                # but for unstable systems, make performance much better.
                next_state = self.prior_dynamics_func(x0=x_var[:, i]-self.prior_ctrl.X_EQ[:,None],
                                                      p=u_var[:, i]-self.prior_ctrl.U_EQ[:,None])['xf'] + \
                             self.prior_ctrl.X_EQ[:,None]+ self.Bd @ self.gaussian_process.casadi_predict(z=z[:,i])['mean']
            opti.subject_to(x_var[:, i + 1] == next_state)
            # Probabilistic state and input constraints according to Hewing 2019 constraint tightening.
            for s_i, state_constraint in enumerate(self.state_constraints_sym):
                if self.gp_soft_constraints:
                    opti.subject_to(state_constraint(x_var[:, i]) <= state_constraint_set[s_i][:,i] + state_slack_list[s_i][:,i])
                    cost += soft_con_coeff*state_slack_list[s_i][:,i].T @ state_slack_list[s_i][:,i]
                    opti.subject_to(state_slack_list[s_i][:,i] >= 0)
                else:
                    opti.subject_to(state_constraint(x_var[:, i]) <= state_constraint_set[s_i][:,i])
            for u_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.gp_soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_constraint_set[u_i][:,i] + input_slack_list[u_i][:, i])
                    cost += soft_con_coeff*input_slack_list[u_i][:,i].T @ input_slack_list[u_i][:,i]
                    opti.subject_to(input_slack_list[u_i][:,i] >= 0)
                else:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_constraint_set[u_i][:, i])

        # Final state constraints.
        for s_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.gp_soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1]) <= state_constraint_set[s_i][:, -1] + state_slack_list[s_i][:,-1])
                cost += soft_con_coeff*state_slack_list[s_i][:,-1].T @ state_slack_list[s_i][:,-1]
                opti.subject_to(state_slack_list[s_i][:,-1] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1]) <= state_constraint_set[s_i][:,-1])

        # Bound constraints.
        upper_state_bounds = np.clip(self.prior_ctrl.env.observation_space.high, None, 10)
        lower_state_bounds = np.clip(self.prior_ctrl.env.observation_space.low, -10, None)
        upper_input_bounds = np.clip(self.prior_ctrl.env.action_space.high, None, 10)
        lower_input_bounds = np.clip(self.prior_ctrl.env.action_space.low, -10, None)
        for i in range(self.T):
            opti.subject_to(opti.bounded(lower_state_bounds, x_var[:,i], upper_state_bounds ))
            opti.subject_to(opti.bounded(lower_input_bounds, u_var[:,i], upper_input_bounds ))
        opti.subject_to(opti.bounded(lower_state_bounds, x_var[:,-1], upper_state_bounds ))

        opti.minimize(cost)
        # Initial condition constraints.
        opti.subject_to(x_var[:, 0] == x_init)
        # Create solver (IPOPT solver in this version).
        opts = {"ipopt.print_level": 4,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 100, #100,
                "print_time": 1,
                "expand": True,
                "verbose": True}
        opti.solver('ipopt', opts)
        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "x_ref": x_ref,
            "state_constraint_set": state_constraint_set,
            "input_constraint_set": input_constraint_set,
            "mean_post_factor": mean_post_factor,
            "z_ind": z_ind,
            "cost": cost,
            "n_ind_points": n_ind_points
        }

        #if False and n_ind_points < self.n_ind_points:
        if not self.sparse_gp:
            mean_post_factor_val, z_ind_val = self.precompute_mean_post_factor_all_data()
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val
        else:
            mean_post_factor_val, Sigma, K_zind_zind_inv, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.mean_post_factor_val = mean_post_factor_val
            #self.Sigma = Sigma
            #self.K_zind_zind_inv = K_zind_zind_inv
            self.z_ind_val = z_ind_val

    def select_action_with_gp(self,
                              obs
                              ):
        """Solves nonlinear MPC problem to get next action.

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
        state_constraint_set = opti_dict["state_constraint_set"]
        input_constraint_set = opti_dict["input_constraint_set"]
        mean_post_factor = opti_dict["mean_post_factor"]
        z_ind = opti_dict["z_ind"]
        cost = opti_dict["cost"]
        n_ind_points = opti_dict["n_ind_points"]
        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.mode == "tracking":
            self.traj_step += 1
        # Set the probabilistic state and input constraint set limits.
        state_constraint_set_prev, input_constraint_set_prev = self.precompute_probabilistic_limits()

        for si in range(len(self.constraints.state_constraints)):
            opti.set_value(state_constraint_set[si], state_constraint_set_prev[si])
        for ui in range(len(self.constraints.input_constraints)):
            opti.set_value(input_constraint_set[ui], input_constraint_set_prev[ui])
        if self.recalc_inducing_points_at_every_step:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.results_dict['inducing_points'].append(z_ind_val)
        else:
            mean_post_factor_val = self.mean_post_factor_val
            z_ind_val = self.z_ind_val
            self.results_dict['inducing_points'] = [z_ind_val]

        opti.set_value(mean_post_factor, mean_post_factor_val)
        opti.set_value(z_ind, z_ind_val)
        # Initial guess for the optimization problem.
        if self.warmstart and self.x_prev is None and self.u_prev is None:
            x_guess, u_guess = self.prior_ctrl.compute_initial_guess(obs, goal_states, self.X_EQ, self.U_EQ)
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)  # Initial guess for optimization problem.
        elif self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-1] = x_guess[:, 1:]
            u_guess[:-1] = u_guess[1:]
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
        except RuntimeError:
            x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
        u_val = np.atleast_2d(u_val)
        self.x_prev = x_val
        self.u_prev = u_val
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['t_wall'].append(opti.stats()['t_wall_total'])
        zi = np.hstack((x_val[:,0], u_val[:,0]))
        zi = zi[self.input_mask]
        gp_contribution = np.sum(self.K_z_zind_func(z1=zi, z2=z_ind_val)['K'].toarray() * mean_post_factor_val,axis=1)
        print("GP Mean eq Contribution: %s" % gp_contribution)
        zi = np.hstack((x_val[:,0], u_val[:,0]))
        pred, _, _ = self.gaussian_process.predict(zi[None,:])
        print("True GP value: %s" % pred.numpy())
        lin_pred = self.prior_dynamics_func(x0=x_val[:,0]-self.prior_ctrl.X_EQ,
                                            p=u_val[:, 0]-self.prior_ctrl.U_EQ)['xf'].toarray() + \
                                            self.prior_ctrl.X_EQ[:,None]
        self.results_dict['linear_pred'].append(lin_pred)
        self.results_dict['gp_mean_eq_pred'].append(gp_contribution)
        self.results_dict['gp_pred'].append(pred.numpy())
        # Take the first one from solved action sequence.
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        self.prev_action = action,
        return action

    def learn(self,
              input_data=None,
              target_data=None,
              gp_model=None,
              overwrite_saved_data: bool = None,
              ):
        """Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.

        Returns:
            training_results (dict): Dictionary of the training results.

        """
        if gp_model is None:
            gp_model = self.gp_model_path
        if overwrite_saved_data is None:
            overwrite_saved_data = self.overwrite_saved_data
        self.prior_ctrl.remove_constraints(self.prior_ctrl.additional_constraints)
        self.reset()
        if self.online_learning:
            input_data = np.zeros((self.train_iterations, len(self.input_mask)))
            target_data = np.zeros((self.train_iterations, len(self.target_mask)))
        if input_data is None and target_data is None:
            # If no input data is provided, we will generate self.train_iterations
            # + (1+self.test_ratio)* self.train_iterations number of training points. This will ensure the specified
            # number of train iterations are run, and the correct train-test data spilt is achieved.
            train_inputs = []
            train_targets = []
            train_info = []

            ############
            # Use Latin Hypercube Sampling to generate states withing environment bounds.
            lhs_sampler = Lhs(lhs_type='classic', criterion='maximin')
            #limits = [(self.env.INIT_STATE_RAND_INFO[key].low, self.env.INIT_STATE_RAND_INFO[key].high) for key in
            #          self.env.INIT_STATE_RAND_INFO]
            limits = [(self.env.INIT_STATE_RAND_INFO['init_'+ key]['low'], self.env.INIT_STATE_RAND_INFO['init_' + key]['high']) for key in self.env.STATE_LABELS]
            # todo: parameterize this if we actually want it.
            num_eq_samples = 0
            validation_iterations = int(self.train_iterations * (self.test_data_ratio / (1 - self.test_data_ratio)))
            samples = lhs_sampler.generate(limits,
                                           self.train_iterations + validation_iterations - num_eq_samples,
                                           random_state=self.seed)
            if self.env.TASK == Task.STABILIZATION and num_eq_samples > 0:
                # todo: choose if we want eq samples or not.
                delta_plus = np.array([0.1, 0.1, 0.1, 0.1, 0.03, 0.3])
                delta_neg = np.array([0.1, 0.1, 0.1, 0.1, 0.03, 0.3])
                eq_limits = [(self.prior_ctrl.env.X_GOAL[eq]-delta_neg[eq], self.prior_ctrl.env.X_GOAL[eq]+delta_plus[eq]) for eq in range(self.model.nx)]
                eq_samples = lhs_sampler.generate(eq_limits, num_eq_samples, random_state=self.seed)
                #samples = samples.append(eq_samples)
                init_state_samples = np.array(samples + eq_samples)
            else:
                init_state_samples = np.array(samples)
            input_limits = np.vstack((self.constraints.input_constraints[0].lower_bounds,
                                      self.constraints.input_constraints[0].upper_bounds)).T
            input_samples = lhs_sampler.generate(input_limits,
                                                 self.train_iterations + validation_iterations,
                                                 random_state=self.seed)
            input_samples = np.array(input_samples) # not being used currently
            seeds = self.env.np_random.randint(0,99999, size=self.train_iterations + validation_iterations)
            for i in range(self.train_iterations + validation_iterations):
                # For random initial state training.
                #init_state = init_state_samples[i,:]
                init_state = dict(zip(self.env.INIT_STATE_RAND_INFO.keys(), init_state_samples[i, :]))
                # Collect data with prior controller.
                run_env = self.env_func(init_state=init_state, randomized_init=False, seed=int(seeds[i]))
                episode_results = self.prior_ctrl.run(env=run_env, max_steps=1)
                run_env.close()
                x_obs = episode_results['obs'][-3:,:]
                u_seq = episode_results['action'][-1:,:]
                run_env.close()
                x_seq = x_obs[:-1,:]
                x_next_seq = x_obs[1:,:]
                train_inputs_i, train_targets_i = self.preprocess_training_data(x_seq, u_seq, x_next_seq)
                train_inputs.append(train_inputs_i)
                train_targets.append(train_targets_i)
            train_inputs = np.vstack(train_inputs)
            train_targets = np.vstack(train_targets)
            self.data_inputs = train_inputs
            self.data_targets = train_targets
        elif input_data is not None and target_data is not None:
            train_inputs = input_data
            train_targets = target_data
            if (self.data_inputs is None and self.data_targets is None) or overwrite_saved_data:
                self.data_inputs = train_inputs
                self.data_targets = train_targets
            else:
                self.data_inputs = np.vstack((self.data_inputs, train_inputs))
                self.data_targets = np.vstack((self.data_targets, train_targets))
        else:
            raise ValueError("[ERROR]: gp_mpc.learn(): Need to provide both targets and inputs.")

        total_input_data = self.data_inputs.shape[0]
        # If validation set is desired.
        if self.test_data_ratio > 0 and self.test_data_ratio is not None:
            train_idx, test_idx = train_test_split(
                                    list(range(total_input_data)),
                                    test_size=self.test_data_ratio,
                                    random_state=self.seed
                                    )

        else:
            # Otherwise, just copy the training data into the test data.
            train_idx = list(range(total_input_data))
            test_idx = list(range(total_input_data))

        train_inputs = self.data_inputs[train_idx, :]
        train_targets = self.data_targets[train_idx, :]
        self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
        test_inputs = self.data_inputs[test_idx, :]
        test_targets = self.data_targets[test_idx, :]
        self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}

        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        # Define likelihood.
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        self.gaussian_process = GaussianProcessCollection(ZeroMeanIndependentGPModel,
                                                     likelihood,
                                                     len(self.target_mask),
                                                     input_mask=self.input_mask,
                                                     target_mask=self.target_mask,
                                                     normalize=self.normalize_training_data
                                                     )
        if gp_model:
            self.gaussian_process.init_with_hyperparam(train_inputs_tensor,
                                                       train_targets_tensor,
                                                       gp_model)
        else:
            # Train the GP.
            self.gaussian_process.train(train_inputs_tensor,
                                        train_targets_tensor,
                                        test_inputs_tensor,
                                        test_targets_tensor,
                                        n_train=self.optimization_iterations,
                                        learning_rate=self.learning_rate,
                                        gpu=self.use_gpu,
                                        dir=self.output_dir)

        self.reset()
        #if self.train_data['train_targets'].shape[0] <= self.n_ind_points:
        #    n_ind_points = self.train_data['train_targets'].shape[0]
        #else:
        #    n_ind_points = self.n_ind_points
        #self.set_gp_dynamics_func(n_ind_points)
        #self.setup_gp_optimizer(n_ind_points)
        self.prior_ctrl.add_constraints(self.prior_ctrl.additional_constraints)
        self.prior_ctrl.reset()
        # Collect training results.
        training_results = {}
        training_results['train_targets'] = train_targets
        training_results['train_inputs'] = train_inputs
        try:
            training_results['info'] = train_info
        except UnboundLocalError:
            training_results['info'] = None
        return training_results

    def select_action(self,
                      obs
                      ):
        """Select the action based on the given observation.

        Args:
            obs (np.array): current observed state.

        Returns:
            action (np.array): desired policy action.

        """
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            if(self.last_obs is not None and self.last_action is not None and self.online_learning):
                print("[ERROR]: Not yet supported.")
                exit()
            t1 = time.perf_counter()
            action = self.select_action_with_gp(obs)
            t2 = time.perf_counter()
            print("GP SELECT ACTION TIME: %s" %(t2 - t1))
            self.last_obs = obs
            self.last_action = action
        return action

    def close(self):
        """Clean up.

        """
        self.env_training.close()
        self.env.close()

    def reset_results_dict(self):
        """

        """
        "Result the results_dict before running."
        super().reset_results_dict()
        self.results_dict['input_constraint_set'] = []
        self.results_dict['state_constraint_set'] = []
        self.results_dict['state_horizon_cov'] = []
        self.results_dict['input_horizon_cov'] = []
        self.results_dict['gp_mean_eq_pred'] = []
        self.results_dict['gp_pred'] = []
        self.results_dict['linear_pred'] = []
        if self.sparse_gp:
            self.results_dict['inducing_points'] = []

    def reset(self):
        """Reset the controller before running.

        """
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        if self.gaussian_process is not None:
            if self.sparse_gp and self.train_data['train_targets'].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data['train_targets'].shape[0]
            elif self.sparse_gp:
                n_ind_points = self.n_ind_points
            else:
                n_ind_points = self.train_data['train_targets'].shape[0]

            self.set_gp_dynamics_func(n_ind_points)
            self.setup_gp_optimizer(n_ind_points)
        self.prior_ctrl.reset()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None
