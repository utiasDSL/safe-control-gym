'''Utility functions for Gaussian Processes.'''

import os.path
from copy import deepcopy

import casadi as ca
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans

from safe_control_gym.utils.utils import mkdirs

torch.manual_seed(0)


def covSEard(x,
             z,
             ell,
             sf2
             ):
    '''GP squared exponential kernel.

    This function is based on the 2018 GP-MPC library by Helge-André Langåker

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): Output scale parameter.

    Returns:
        SE kernel (casadi.MX/SX): Squared Exponential kernel.
    '''
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.SX.exp(-.5 * dist)


def covMatern52ard(x,
                   z,
                   ell,
                   sf2
                   ):
    '''Matern kernel that takes nu equal to 5/2.

    Args:
        x (np.array or casadi.MX/SX): First vector.
        z (np.array or casadi.MX/SX): Second vector.
        ell (np.array or casadi.MX/SX): Length scales.
        sf2 (float or casadi.MX/SX): Output scale parameter.

    Returns:
        Matern52 kernel (casadi.MX/SX): Matern 5/2 kernel.
    '''
    dist = ca.sum1((x - z)**2 / ell**2)
    r_over_l = ca.sqrt(dist)
    return sf2 * (1 + ca.sqrt(5) * r_over_l + 5 / 3 * r_over_l ** 2) * ca.exp(- ca.sqrt(5) * r_over_l)


class ZeroMeanIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    '''Multidimensional Gaussian Process model with zero mean function,
       or constant mean and radial basis function kernel (SE).
    '''

    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 nx,
                 kernel='RBF'
                 ):
        '''Initialize a multidimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): Input training data (input_dim X N samples).
            train_y (torch.Tensor): Output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.MultitaskGaussianLikelihood).
            nx (int): Dimension of the target output (output dim).
        '''
        super().__init__(train_x, train_y, likelihood)
        self.n = nx
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([self.n]))
        # For constant mean function.
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.n]),
                                           ard_num_dims=train_x.shape[1]),
                batch_shape=torch.Size([self.n]),
                ard_num_dims=train_x.shape[1]
            )
        elif kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(batch_shape=torch.Size([self.n]),
                                              ard_num_dims=train_x.shape[1]),
                batch_shape=torch.Size([self.n]),
                ard_num_dims=train_x.shape[1]
            )
        else:
            raise NotImplementedError

    def forward(self,
                x
                ):
        '''Forward pass for the GP model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            gpytorch.distributions.MultitaskMultivariateNormal: Multitask GP output.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class ZeroMeanIndependentGPModel(gpytorch.models.ExactGP):
    '''Single dimensional output Gaussian Process model with zero mean function,
       or constant mean and radial basis function kernel (SE).
    '''

    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 kernel='RBF'
                 ):
        '''Initialize a single dimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): Input training data (input_dim X N samples).
            train_y (torch.Tensor): Output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.GaussianLikelihood).
            kernel (str): Kernel type, 'RBF' or 'Matern'.
        '''
        super().__init__(train_x, train_y, likelihood)
        # For Zero mean function.
        self.mean_module = gpytorch.means.ZeroMean()
        # For constant mean function.
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
                ard_num_dims=train_x.shape[1]
            )
        elif kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]),
                ard_num_dims=train_x.shape[1]
            )

    def forward(self,
                x
                ):
        '''Forward pass for the GP model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: GP output.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    '''Multidimensional Gaussian Process model with zero mean function.
    '''

    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 kernel='RBF'
                 ):
        '''Initialize a multidimensional Gaussian Process model with zero mean function.

        Args:
            train_x (torch.Tensor): Input training data (input_dim X N samples).
            train_y (torch.Tensor): Output training data (output dim x N samples).
            likelihood (gpytorch.likelihood): Likelihood function (gpytorch.likelihoods.GaussianLikelihood with batch).
            kernel (str): Kernel type, 'RBF' or 'Matern'.
        '''
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([train_y.shape[0]]))
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([train_y.shape[0]])),
                batch_shape=torch.Size([train_y.shape[0]]), ard_num_dims=train_x.shape[-1]
            )
        elif kernel == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[-1], batch_shape=torch.Size([train_y.shape[0]])),
                batch_shape=torch.Size([train_y.shape[0]]), ard_num_dims=train_x.shape[-1]
            )

    def forward(self, x):
        '''Forward pass for the batch multitask GP model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: GP output.
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessCollection:
    '''Collection of Gaussian Processes for multi-output GPs.'''

    def __init__(self, model_type,
                 likelihood,
                 target_dim,
                 input_mask=None,
                 target_mask=None,
                 normalize=False,
                 kernel='Matern',
                 parallel=False
                 ):
        '''Creates a single GaussianProcess for each output dimension.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
            target_dim (int): Dimension of the output (how many GPs to make).
            input_mask (list): Input dimensions to keep. If None, use all input dimensions.
            target_mask (list): Target dimensions to keep. If None, use all target dimensions.
            normalize (bool): If True, scale all data between -1 and 1.
        '''
        self.gp_list = []
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.NORMALIZE = normalize
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.parallel = parallel
        if parallel:
            self.gps = BatchGPModel(model_type,
                                    likelihood,
                                    input_mask=input_mask,
                                    target_mask=target_mask,
                                    normalize=normalize,
                                    kernel=kernel)
        else:
            for _ in range(target_dim):
                self.gp_list.append(GaussianProcess(model_type,
                                                    deepcopy(likelihood),
                                                    input_mask=input_mask,
                                                    normalize=normalize,
                                                    kernel=kernel))

    def _init_properties(self,
                         train_inputs,
                         train_targets
                         ):
        '''Initialize useful properties.

        Args:
            train_inputs (torch.Tensor): Input training data.
            train_targets (torch.Tensor): Target training data.
        '''
        target_dimension = train_targets.shape[1]
        self.input_dimension = train_inputs.shape[1]
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def init_with_hyperparam(self,
                             train_inputs,
                             train_targets,
                             path_to_statedicts
                             ):
        '''Load hyperparameters from a state_dict.

        Args:
            train_inputs (torch.Tensor): Input training data.
            train_targets (torch.Tensor): Target training data.
            path_to_statedicts (str): Path to where the state dicts are saved.
        '''
        assert self.parallel is False, ValueError('Parallel GP not supported yet.')

        self._init_properties(train_inputs, train_targets)
        gp_K_plus_noise_list = []
        gp_K_plus_noise_inv_list = []
        for gp_ind, gp in enumerate(self.gp_list):
            path = os.path.join(path_to_statedicts, f'best_model_{self.target_mask[gp_ind]}.pth')
            print('#########################################')
            print('#       Loading GP dimension {self.target_mask[gp_ind]}         #')
            print('#########################################')
            print(f'Path: {path}')
            gp.init_with_hyperparam(train_inputs,
                                    train_targets[:, self.target_mask[gp_ind]],
                                    path)
            gp_K_plus_noise_list.append(gp.model.K_plus_noise.detach())
            gp_K_plus_noise_inv_list.append(gp.model.K_plus_noise_inv.detach())
            print('Loaded!')
        gp_K_plus_noise = torch.stack(gp_K_plus_noise_list)
        gp_K_plus_noise_inv = torch.stack(gp_K_plus_noise_inv_list)
        self.K_plus_noise = gp_K_plus_noise
        self.K_plus_noise_inv = gp_K_plus_noise_inv
        self.casadi_predict = self.make_casadi_predict_func()

    def get_hyperparameters(self,
                            as_numpy=False
                            ):
        '''Get the output scale and length scale from the kernel matrices of the GPs.

        Args:
            as_numpy (bool): If True, return as numpy arrays.

        Returns:
            lengthscale (torch.Tensor): Length scale of the GPs.
            outputscale (torch.Tensor): Output scale of the GPs.
            noise (torch.Tensor): Noise of the GPs.
            K_plus_noise (torch.Tensor): Kernel matrix of the GPs.
        '''
        lengthscale_list = []
        output_scale_list = []
        noise_list = []
        if self.parallel is False:
            for gp in self.gp_list:
                lengthscale_list.append(gp.model.covar_module.base_kernel.lengthscale.detach())
                output_scale_list.append(gp.model.covar_module.outputscale.detach())
                noise_list.append(gp.model.likelihood.noise.detach())
            lengthscale = torch.cat(lengthscale_list)
            outputscale = torch.Tensor(output_scale_list)
            noise = torch.Tensor(noise_list)
        else:
            lengthscale = self.gps.model.covar_module.base_kernel.lengthscale.detach().squeeze()
            outputscale = self.gps.model.covar_module.outputscale.detach()
            noise = self.gps.model.likelihood.noise.detach()

        if as_numpy:
            return lengthscale.numpy(), outputscale.numpy(), noise.numpy(), self.K_plus_noise.detach().numpy()
        else:
            return lengthscale, outputscale, noise, self.K_plus_noise

    def train(self,
              train_x_raw,
              train_y_raw,
              test_x_raw,
              test_y_raw,
              n_train=[500],
              learning_rate=[0.01],
              gpu=False,
              output_dir='results'
              ):
        '''Train the GP using train_x and train_y.

        Args:
            train_x_raw (torch.Tensor): Training input data.
            train_y_raw (torch.Tensor): Training target data.
            test_x_raw (torch.Tensor): Test input data.
            test_y_raw (torch.Tensor): Test target data.
            n_train (list): Number of training iterations for each GP.
            learning_rate (list): Learning rate for each GP.
            gpu (bool): If True, use GPU.
            output_dir (str): Directory to save models.
        '''
        self._init_properties(train_x_raw, train_y_raw)
        self.model_paths = []
        mkdirs(output_dir)

        if self.parallel is False:
            gp_K_plus_noise_inv_list = []
            gp_K_plus_noise_list = []
            for gp_ind, gp in enumerate(self.gp_list):
                lr = learning_rate[self.target_mask[gp_ind]]
                n_t = n_train[self.target_mask[gp_ind]]
                print('#########################################')
                print(f'#      Training GP dimension {self.target_mask[gp_ind]}         #')
                print('#########################################')
                print(f'Train iterations: {n_t}')
                print(f'Learning Rate: {lr}')
                gp.train(train_x_raw,
                         train_y_raw[:, self.target_mask[gp_ind]],
                         test_x_raw,
                         test_y_raw[:, self.target_mask[gp_ind]],
                         n_train=n_t,
                         learning_rate=lr,
                         gpu=gpu,
                         fname=os.path.join(output_dir, f'best_model_{self.target_mask[gp_ind]}.pth'))
                self.model_paths.append(output_dir)
                gp_K_plus_noise_list.append(gp.model.K_plus_noise)
                gp_K_plus_noise_inv_list.append(gp.model.K_plus_noise_inv)
            gp_K_plus_noise = torch.stack(gp_K_plus_noise_list)
            gp_K_plus_noise_inv = torch.stack(gp_K_plus_noise_inv_list)
            self.K_plus_noise = gp_K_plus_noise
            self.K_plus_noise_inv = gp_K_plus_noise_inv
        else:
            for i in range(len(learning_rate)):
                assert learning_rate[i] == learning_rate[0], ValueError('Learning rate must be the same for all GPs.')
            for i in range(len(n_train)):
                assert n_train[i] == n_train[0], ValueError('Training iterations must be the same for all GPs.')
            lr = learning_rate[0]
            n_t = n_train[0]
            self.gps.train(train_x_raw,
                           train_y_raw,
                           test_x_raw,
                           test_y_raw,
                           n_train=n_t,
                           learning_rate=lr,
                           gpu=gpu,
                           fname=os.path.join(output_dir, 'best_model.pth'))
            self.K_plus_noise = self.gps.gp_K_plus_noise
            self.K_plus_noise_inv = self.gps.gp_K_plus_noise_inv

        self.casadi_predict = self.make_casadi_predict_func()

    def predict(self,
                x,
                requires_grad=False,
                return_pred=True
                ):
        '''Predict using the GP.

        Args:
            x (torch.Tensor): Input data (N_samples x input_dim).
            requires_grad (bool): If True, compute gradients.
            return_pred (bool): If True, return prediction object.

        Returns:
            means (torch.Tensor): Mean of the GP.
            cov (torch.Tensor): Covariance of the GP.
            pred_list (list): List of predictions.
        '''
        if self.parallel is False:
            means_list = []
            cov_list = []
            pred_list = []
            for gp in self.gp_list:
                if return_pred:
                    mean, cov, pred = gp.predict(x, requires_grad=requires_grad, return_pred=return_pred)
                    pred_list.append(pred)
                else:
                    mean, cov = gp.predict(x, requires_grad=requires_grad, return_pred=return_pred)
                means_list.append(mean)
                cov_list.append(cov)
            means = torch.tensor(means_list)
            cov = torch.diag(torch.cat(cov_list).squeeze())
            if return_pred:
                return means, cov, pred_list
            else:
                return means, cov
        else:
            return self.gps.predict(x, requires_grad=requires_grad, return_pred=return_pred)

    def make_casadi_predict_func(self):
        '''Create a CasADi function for GP prediction.
        Assumes train_inputs and train_targets are already masked.

        Returns:
            casadi_predict (casadi.Function): CasADi prediction function.
        '''

        Nz = len(self.input_mask)
        Ny = len(self.target_mask)
        z = ca.SX.sym('z1', Nz)
        y = ca.SX.zeros(Ny)

        if self.parallel is False:
            for gp_ind, gp in enumerate(self.gp_list):
                y[gp_ind] = gp.casadi_predict(z=z)['mean']
        else:
            for i in range(Ny):
                y[i] = self.gps.casadi_predict[i](z=z)['mean']

        casadi_predict = ca.Function('pred',
                                     [z],
                                     [y],
                                     ['z'],
                                     ['mean'])
        return casadi_predict

    def prediction_jacobian(self,
                            query
                            ):
        '''Return the Jacobian of the GP prediction.

        Args:
            query (torch.Tensor): Query input.

        Returns:
            jacobian (torch.Tensor): Jacobian of the mean prediction.
        '''
        raise NotImplementedError

    def plot_trained_gp(self,
                        inputs,
                        targets,
                        fig_count=0
                        ):
        '''Plot the trained GP given the input and target data.

        Args:
            inputs (np.array): Input data.
            targets (np.array): Target data.
            fig_count (int): Figure count for plotting.
        '''
        assert self.parallel is False, ValueError('Parallel GP not supported yet.')
        for gp_ind, gp in enumerate(self.gp_list):
            fig_count = gp.plot_trained_gp(inputs,
                                           targets[:, self.target_mask[gp_ind], None],
                                           self.target_mask[gp_ind],
                                           fig_count=fig_count)
            fig_count += 1

    def _kernel_list(self,
                     x1,
                     x2=None
                     ):
        '''Evaluate the kernel given vectors x1 and x2.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            k_list (list): List of LazyTensor kernels.
        '''
        if x2 is None:
            x2 = x1
        # TODO: Make normalization at the GPCollection level?
        # if self.NORMALIZE:
        #    x1 = torch.from_numpy(self.gp_list[0].scaler.transform(x1.numpy()))
        #    x2 = torch.from_numpy(self.gp_list[0].scaler.transform(x2.numpy()))
        k_list = []
        if self.parallel is False:
            for gp in self.gp_list:
                k_list.append(gp.model.covar_module(x1, x2))
        else:
            k_list = list(self.gps.model.covar_module(x1, x2))

        return k_list

    def kernel(self,
               x1,
               x2=None
               ):
        '''Evaluate the kernel given vectors x1 and x2.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            non_lazy_tensors (torch.Tensor): Non-lazy kernel matrices.
        '''
        k_list = self._kernel_list(x1, x2)
        non_lazy_tensors = [k.evaluate() for k in k_list]
        return torch.stack(non_lazy_tensors)

    def kernel_inv(self,
                   x1,
                   x2=None
                   ):
        '''Evaluate the inverse kernel given vectors x1 and x2.

        Only works for square kernel.

        Args:
            x1 (torch.Tensor): First vector.
            x2 (torch.Tensor): Second vector.

        Returns:
            non_lazy_tensors (torch.Tensor): Non-lazy inverse kernel matrices.
        '''
        if x2 is None:
            x2 = x1
        assert x1.shape == x2.shape, ValueError('x1 and x2 need to have the same shape.')
        k_list = self._kernel_list(x1, x2)
        num_of_points = x1.shape[0]
        # Efficient inversion is performed VIA inv_matmul on the laze tensor with Identity.
        non_lazy_tensors = [k.inv_matmul(torch.eye(num_of_points).double()) for k in k_list]
        return torch.stack(non_lazy_tensors)


class BatchGPModel:
    '''Gaussian Processes decorator for batch GP in gpytorch.'''

    def __init__(self,
                 model_type,
                 likelihood,
                 input_mask=None,
                 target_mask=None,
                 normalize=False,
                 kernel='RBF',
                 ):
        '''Initialize Gaussian Process.

        Args:
            model_type (gpytorch model class): Model class for the GP (BatchIndependentMultitaskGPModel).
            likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood): likelihood function.
            normalize (bool): If True, scale all data between -1 and 1. (prototype and not fully operational).
        '''
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.NORMALIZE = normalize
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.kernel = kernel
        assert normalize is False, NotImplementedError('Normalization not implemented yet.')

    def _init_model(self,
                    train_inputs,
                    train_targets
                    ):
        '''Initialize GP model from train inputs and train_targets.

        Args:
            train_inputs (torch.Tensor): Input training data.
            train_targets (torch.Tensor): Target training data.
        '''
        if train_targets.ndim > 1:
            target_dimension = train_targets.shape[1]
        else:
            target_dimension = 1

        self.likelihood = self.likelihood
        self.model = BatchIndependentMultitaskGPModel(train_inputs.unsqueeze(0).repeat(target_dimension, 1, 1), train_targets.T, self.likelihood, kernel=self.kernel)

        # Extract dimensions for external use.
        self.input_dimension = train_inputs.shape[1]
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def _compute_GP_covariances(self,
                                train_x
                                ):
        '''Compute K(X,X) + sigma*I and its inverse.

        Args:
            train_x (torch.Tensor): Training input data.
        '''
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K_lazy = self.model.covar_module(train_x)
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.gp_K_plus_noise = K_lazy_plus_noise.matmul(torch.eye(n_samples).double())
        self.gp_K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())

    def init_with_hyperparam(self,
                             train_inputs,
                             train_targets,
                             path_to_statedict
                             ):
        '''Load hyperparameters from a state_dict.

        Args:
            train_inputs (torch.Tensor): Input training data.
            train_targets (torch.Tensor): Target training data.
            path_to_statedict (str): Path to the state dict.
        '''

        if self.input_mask is not None:
            train_inputs = train_inputs[:, self.input_mask]
        if self.target_mask is not None:
            train_targets = train_targets[:, self.target_mask]
        device = torch.device('cpu')
        state_dict = torch.load(path_to_statedict, map_location=device, _use_new_zipfile_serialization=True)
        self._init_model(train_inputs, train_targets)

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(dtype=torch.float64)  # needed otherwise loads state_dict as float32
        self._compute_GP_covariances(train_inputs)
        self.casadi_predict = self.make_casadi_prediction_func(train_inputs, train_targets)

    def train(self,
              train_input_data,
              train_target_data,
              test_input_data,
              test_target_data,
              n_train=500,
              learning_rate=0.01,
              gpu=False,
              fname='best_model.pth',
              ):
        '''Train the GP using train_x and train_y.

        Args:
            train_input_data (torch.Tensor): Training input data.
            train_target_data (torch.Tensor): Training target data.
            test_input_data (torch.Tensor): Test input data.
            test_target_data (torch.Tensor): Test target data.
            n_train (int): Number of training iterations.
            learning_rate (float): Learning rate.
            gpu (bool): If True, use GPU.
            fname (str): File name to save the model.
        '''
        train_x_raw = train_input_data
        train_y_raw = train_target_data
        test_x_raw = test_input_data
        test_y_raw = test_target_data
        if self.input_mask is not None:
            train_x_raw = train_x_raw[:, self.input_mask]
            test_x_raw = test_x_raw[:, self.input_mask]
        if self.target_mask is not None:
            train_y_raw = train_y_raw[:, self.target_mask]
            test_y_raw = test_y_raw[:, self.target_mask]
        self._init_model(train_x_raw, train_y_raw)

        train_x = train_x_raw
        train_y = train_y_raw
        test_x = test_x_raw
        test_y = test_y_raw

        if gpu:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        self.model = self.model.to(dtype=torch.float64)
        self.likelihood = self.likelihood.to(dtype=torch.float64)
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        last_loss = 99999999
        best_loss = 99999999
        loss = torch.tensor(0)
        i = 0
        while i < n_train and torch.abs(loss - last_loss) > 1e-2:
            with torch.inference_mode():
                self.model.eval()
                self.likelihood.eval()
                test_output = self.model(test_x.unsqueeze(0).repeat(self.output_dimension, 1, 1))
                test_loss = -mll(test_output, test_y.T).sum()
            self.model.train()
            self.likelihood.train()
            self.optimizer.zero_grad()
            output = self.model(train_x.unsqueeze(0).repeat(self.output_dimension, 1, 1))
            loss = -mll(output, train_y.T).sum()
            loss.backward()
            if i % 100 == 0:
                print('Iter %d/%d - MLL train Loss: %.3f, Posterior Test Loss: %0.3f' % (i + 1, n_train, loss.item(), test_loss.item()))

            self.optimizer.step()

            if test_loss < best_loss:
                best_loss = test_loss
                state_dict = self.model.state_dict()
                torch.save(state_dict, fname, _use_new_zipfile_serialization=True)
                best_epoch = i

            i += 1
        print('Training Complete')
        print('Lowest epoch: %s' % best_epoch)
        print('Lowest Loss: %s' % best_loss)
        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        self.model.load_state_dict(torch.load(fname, weights_only=False))
        self._compute_GP_covariances(train_x)
        self.casadi_predict = self.make_casadi_prediction_func(train_x, train_y)

        return

    def predict(self,
                x,
                requires_grad=False,
                return_pred=True
                ):
        '''Predict using the GP.

        Args:
            x (torch.Tensor or np.ndarray): Input data (N_samples x input_dim).
            requires_grad (bool): If True, compute gradients.
            return_pred (bool): If True, return prediction object.

        Returns:
            means (torch.Tensor): Mean of the GP.
            cov (torch.Tensor): Covariance of the GP.
            predictions (torch.Tensor): Predictions of the GP.
        '''
        self.model.eval()
        self.likelihood.eval()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if self.input_mask is not None:
            x = x[:, self.input_mask]
        if requires_grad:
            predictions = self.likelihood(self.model(x.unsqueeze(0).repeat(self.output_dimension, 1, 1)))
            means = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(state=True), gpytorch.settings.fast_pred_samples(state=True):
                predictions = self.likelihood(self.model(x.unsqueeze(0).repeat(self.output_dimension, 1, 1)))
                means = predictions.mean
                cov = predictions.covariance_matrix

        means = means.squeeze()
        cov = torch.diag(cov.squeeze())

        if return_pred:
            return means, cov, predictions
        else:
            return means, cov

    def prediction_jacobian(self,
                            query
                            ):
        '''Return the Jacobian of the GP prediction.

        Args:
            query (torch.Tensor): Query input.

        Returns:
            mean_der (torch.Tensor): Jacobian of the mean prediction.
        '''
        mean_der, cov_der = torch.autograd.functional.jacobian(
            lambda x: self.predict(x, requires_grad=True, return_pred=False),
            query.double())
        return mean_der.detach().squeeze()

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        '''Create a CasADi function for GP prediction.

        Args:
            train_inputs (torch.Tensor): Training input data.
            train_targets (torch.Tensor): Training target data.

        Returns:
            y (list): List of CasADi prediction functions.
        '''
        Nx = len(self.input_mask)
        Ny = len(self.target_mask)
        z = ca.SX.sym('z', Nx)
        y = []

        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()

        lengthscales = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scales = self.model.covar_module.outputscale.detach().numpy()

        for i in range(Ny):
            lengthscale = lengthscales[i]
            output_scale = output_scales[i]
            z = ca.SX.sym('z', Nx)
            if self.kernel == 'RBF':
                K_z_ztrain = ca.Function('k_z_ztrain',
                                         [z],
                                         [covSEard(z, train_inputs.T, lengthscale.T, output_scale)],
                                         ['z'],
                                         ['K'])
            elif self.kernel == 'Matern':
                K_z_ztrain = ca.Function('k_z_ztrain',
                                         [z],
                                         [covMatern52ard(z, train_inputs.T, lengthscale.T, output_scale)],
                                         ['z'],
                                         ['K'])
            y += [ca.Function('pred',
                              [z],
                              [K_z_ztrain(z=z)['K'] @ self.gp_K_plus_noise_inv[i, :, :].detach().numpy() @ train_targets[:, i]],
                              ['z'],
                              ['mean'])]

        return y

    def plot_trained_gp(self,
                        inputs,
                        targets,
                        output_label,
                        fig_count=0
                        ):
        '''Plot the trained GP for a given output label.

        Args:
            inputs (np.array): Input data.
            targets (np.array): Target data.
            output_label (int): Output label index.
            fig_count (int): Figure count for plotting.

        Returns:
            fig_count (int): Updated figure count.
        '''
        raise NotImplementedError


class GaussianProcess:
    '''Gaussian Process decorator for gpytorch.'''

    def __init__(self,
                 model_type,
                 likelihood,
                 input_mask=None,
                 target_mask=None,
                 normalize=False,
                 kernel='RBF',
                 ):
        '''Initialize Gaussian Process.

        Args:
            model_type (gpytorch model class): Model class for the GP (ZeroMeanIndependentMultitaskGPModel).
            likelihood (gpytorch.likelihood): likelihood function.
            input_mask (list): List of input mask indices.
            target_mask (list): List of target mask indices.
            normalize (bool): If True, scale all data between -1 and 1. (prototype and not fully operational).
            kernel (str): Kernel type, 'RBF' or 'Matern'.
        '''
        self.model_type = model_type
        self.likelihood = likelihood
        self.optimizer = None
        self.model = None
        self.NORMALIZE = normalize
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.kernel = kernel

    def _init_model(self,
                    train_inputs,
                    train_targets
                    ):
        '''Initialize GP model from train inputs and train_targets.

        Args:
            train_inputs (torch.Tensor): Input training data.
            train_targets (torch.Tensor): Target training data.
        '''
        if train_targets.ndim > 1:
            target_dimension = train_targets.shape[1]
        else:
            target_dimension = 1

        # Define normalization scaler.
        self.scaler = preprocessing.StandardScaler().fit(train_inputs.numpy())
        if self.NORMALIZE:
            train_inputs = torch.from_numpy(self.scaler.transform(train_inputs.numpy()))

        if self.model is None:
            self.model = self.model_type(train_inputs,
                                         train_targets,
                                         self.likelihood,
                                         self.kernel)
        # Extract dimensions for external use.
        self.input_dimension = train_inputs.shape[1]
        self.output_dimension = target_dimension
        self.n_training_samples = train_inputs.shape[0]

    def _compute_GP_covariances(self,
                                train_x
                                ):
        '''Compute K(X,X) + sigma*I and its inverse.

        Args:
            train_x (torch.Tensor): Training input data.
        '''
        # Pre-compute inverse covariance plus noise to speed-up computation.
        K_lazy = self.model.covar_module(train_x.double())
        K_lazy_plus_noise = K_lazy.add_diag(self.model.likelihood.noise)
        n_samples = train_x.shape[0]
        self.model.K_plus_noise = K_lazy_plus_noise.matmul(torch.eye(n_samples).double())
        self.model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())
        # self.model.K_plus_noise_inv_2 = torch.inverse(self.model.K_plus_noise) # Equivalent to above but slower.

    def init_with_hyperparam(self,
                             train_inputs,
                             train_targets,
                             path_to_statedict
                             ):
        '''Load hyperparameters from a state_dict.

        Args:
            train_inputs (torch.Tensor): Input training data.
            train_targets (torch.Tensor): Target training data.
            path_to_statedict (str): Path to the state dict.
        '''
        if self.input_mask is not None:
            train_inputs = train_inputs[:, self.input_mask]
        if self.target_mask is not None:
            train_targets = train_targets[:, self.target_mask]
        device = torch.device('cpu')
        state_dict = torch.load(path_to_statedict, map_location=device, _use_new_zipfile_serialization=True)
        self._init_model(train_inputs, train_targets)
        if self.NORMALIZE:
            train_inputs = torch.from_numpy(self.scaler.transform(train_inputs.numpy()))
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(dtype=torch.float64)  # needed otherwise loads state_dict as float32
        self._compute_GP_covariances(train_inputs)
        self.casadi_predict = self.make_casadi_prediction_func(train_inputs, train_targets)

    def train(self,
              train_input_data,
              train_target_data,
              test_input_data,
              test_target_data,
              n_train=500,
              learning_rate=0.01,
              gpu=False,
              fname='best_model.pth',
              ):
        '''Train the GP using train_x and train_y.

        Args:
            train_input_data (torch.Tensor): Training input data.
            train_target_data (torch.Tensor): Training target data.
            test_input_data (torch.Tensor): Test input data.
            test_target_data (torch.Tensor): Test target data.
            n_train (int): Number of training iterations.
            learning_rate (float): Learning rate.
            gpu (bool): If True, use GPU.
            fname (str): File name to save the model.
        '''
        train_x_raw = train_input_data
        train_y_raw = train_target_data
        test_x_raw = test_input_data
        test_y_raw = test_target_data
        if self.input_mask is not None:
            train_x_raw = train_x_raw[:, self.input_mask]
            test_x_raw = test_x_raw[:, self.input_mask]
        if self.target_mask is not None:
            train_y_raw = train_y_raw[:, self.target_mask]
            test_y_raw = test_y_raw[:, self.target_mask]
        self._init_model(train_x_raw, train_y_raw)
        if self.NORMALIZE:
            train_x = torch.from_numpy(self.scaler.transform(train_x_raw))
            test_x = torch.from_numpy(self.scaler.transform(test_x_raw))
            train_y = train_y_raw
            test_y = test_y_raw
        else:
            train_x = train_x_raw
            train_y = train_y_raw
            test_x = test_x_raw
            test_y = test_y_raw
        if gpu:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        self.model = self.model.to(dtype=torch.float64)
        self.likelihood = self.likelihood.to(dtype=torch.float64)
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        last_loss = 99999999
        best_loss = 99999999
        loss = torch.tensor(0)
        i = 0
        while i < n_train and torch.abs(loss - last_loss) > 1e-2:
            with torch.inference_mode():
                self.model.eval()
                self.likelihood.eval()
                test_output = self.model(test_x)
                test_loss = -mll(test_output, test_y)
            self.model.train()
            self.likelihood.train()
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if i % 100 == 0:
                print('Iter %d/%d - MLL train Loss: %.3f, Posterior Test Loss: %0.3f' % (i + 1, n_train, loss.item(), test_loss.item()))

            self.optimizer.step()
            if test_loss < best_loss:
                best_loss = test_loss
                state_dict = self.model.state_dict()
                torch.save(state_dict, fname, _use_new_zipfile_serialization=True)
                best_epoch = i

            i += 1
        print('Training Complete')
        print(f'Lowest epoch: {best_epoch}')
        print(f'Lowest Loss: {best_loss}')
        self.model = self.model.cpu()
        self.likelihood = self.likelihood.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        self.model.load_state_dict(torch.load(fname, weights_only=False))
        self._compute_GP_covariances(train_x)
        self.casadi_predict = self.make_casadi_prediction_func(train_x, train_y)

    def predict(self,
                x,
                requires_grad=False,
                return_pred=True
                ):
        '''Predict using the GP.

        Args:
            x (torch.Tensor or np.ndarray): Input data (N_samples x input_dim).
            requires_grad (bool): If True, compute gradients.
            return_pred (bool): If True, return prediction object.

        Returns:
            mean (torch.Tensor): Mean of the GP.
            cov (torch.Tensor): Covariance of the GP.
            predictions (torch.Tensor): Predictions of the GP.
        '''
        self.model.eval()
        self.likelihood.eval()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if self.input_mask is not None:
            x = x[:, self.input_mask]
        if self.NORMALIZE:
            x = torch.from_numpy(self.scaler.transform(x))
        if requires_grad:
            predictions = self.likelihood(self.model(x))
            mean = predictions.mean
            cov = predictions.covariance_matrix
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var(state=True), gpytorch.settings.fast_pred_samples(state=True):
                predictions = self.likelihood(self.model(x))
                mean = predictions.mean
                cov = predictions.covariance_matrix
        if return_pred:
            return mean, cov, predictions
        else:
            return mean, cov

    def prediction_jacobian(self,
                            query
                            ):
        '''Return the Jacobian of the GP prediction.

        Args:
            query (torch.Tensor): Query input.

        Returns:
            mean_der (torch.Tensor): Jacobian of the mean prediction.
        '''
        mean_der, _ = torch.autograd.functional.jacobian(
            lambda x: self.predict(x, requires_grad=True, return_pred=False),
            query.double())
        return mean_der.detach().squeeze()

    def make_casadi_prediction_func(self, train_inputs, train_targets):
        '''Create a CasADi function for GP prediction.
           Assumes train_inputs and train_targets are already masked.

        Args:
            train_inputs (torch.Tensor): Training input data.
            train_targets (torch.Tensor): Training target data.

        Returns:
            predict (casadi.Function): CasADi prediction function.
        '''
        train_inputs = train_inputs.numpy()
        train_targets = train_targets.numpy()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
        output_scale = self.model.covar_module.outputscale.detach().numpy()
        Nx = len(self.input_mask)
        z = ca.SX.sym('z', Nx)
        if self.kernel == 'RBF':
            K_z_ztrain = ca.Function('k_z_ztrain',
                                     [z],
                                     [covSEard(z, train_inputs.T, lengthscale.T, output_scale)],
                                     ['z'],
                                     ['K'])
        elif self.kernel == 'Matern':
            K_z_ztrain = ca.Function('k_z_ztrain',
                                     [z],
                                     [covMatern52ard(z, train_inputs.T, lengthscale.T, output_scale)],
                                     ['z'],
                                     ['K'])
        predict = ca.Function('pred',
                              [z],
                              [K_z_ztrain(z=z)['K'] @ self.model.K_plus_noise_inv.detach().numpy() @ train_targets],
                              ['z'],
                              ['mean'])
        return predict

    def plot_trained_gp(self,
                        inputs,
                        targets,
                        output_label,
                        fig_count=0
                        ):
        '''Plot the trained GP for a given output label.

        Args:
            inputs (np.array): Input data.
            targets (np.array): Target data.
            output_label (int): Output label index.
            fig_count (int): Figure count for plotting.

        Returns:
            fig_count (int): Updated figure count.
        '''
        if self.target_mask is not None:
            targets = targets[:, self.target_mask]
        means, _, preds = self.predict(inputs)
        t = np.arange(inputs.shape[0])
        lower, upper = preds.confidence_region()
        for i in range(self.output_dimension):
            fig_count += 1
            plt.figure(fig_count)
            if lower.ndim > 1:
                plt.fill_between(t, lower[:, i].detach().numpy(), upper[:, i].detach().numpy(), alpha=0.5, label='95%')
                plt.plot(t, means[:, i], 'r', label='GP Mean')
                plt.plot(t, targets[:, i], '*k', label='Data')
            else:
                plt.fill_between(t, lower.detach().numpy(), upper.detach().numpy(), alpha=0.5, label='95%')
                plt.plot(t, means, 'r', label='GP Mean')
                plt.plot(t, targets, '*k', label='Data')
            plt.legend()
            plt.title(f'Fitted GP x{output_label}')
            plt.xlabel('Time (s)')
            plt.ylabel('v')
            plt.show()
        return fig_count


def kmeans_centriods(n_cent, data, rand_state=0):
    '''KMeans clustering. Useful for finding reasonable inducing points.

    Args:
        n_cent (int): Number of centroids.
        data (np.array): Data to find the centroids of (n_samples X n_features).
        rand_state (int): Random state for reproducibility.

    Returns:
        centroids (np.array): Array of centroids (n_cent X n_features).
    '''
    kmeans = KMeans(n_clusters=n_cent, random_state=rand_state).fit(data)
    return kmeans.cluster_centers_
