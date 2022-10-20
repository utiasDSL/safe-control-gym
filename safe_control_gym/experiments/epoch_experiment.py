'''Epoch based training for experiments. '''
import numpy as np
from copy import deepcopy

from safe_control_gym.utils.utils import is_wrapped
from safe_control_gym.experiments.base_experiment import BaseExperiment, RecordDataWrapper, MetricExtractor
from safe_control_gym.math_and_models.metrics.performance_metrics import compute_cvar


class EpochExp(BaseExperiment):
    def __init__(self,
                 test_envs,
                 ctrl,
                 train_envs,
                 n_epochs,
                 n_train_episodes_per_epoch,
                 n_test_episodes_per_epoch,
                 output_dir,
                 save_train_trajs: bool=False,
                 save_test_trajs: bool=False
                 ):
        """ Class for an experiment using Epoch training with evaluation every iteration.

        Args:
            test_envs (list of BenchmarkEnv): List of evaluation environments. There should be either one per epoch, or only
                a single evaluation environment. If the controller has a prior controller, there should be n_epochs + 1
                test envs (one for the prior controller).
            ctrl (BaseController): The controller being used.
            train_envs (list of BenchmarkEnvs): The environments being used. There shoud be either one per epoch, or
                only a single test environment.
            n_epochs (int): The number of epochs.
            n_train_episodes_per_epoch (int): The number of train episodes to run per epoch.
            n_test_episodes_per_epoch (int): The number of test episodes to run per epoch.
            output_dir (str): Output dir to store any saved data.
            save_train_trajs (bool): Save all training trajectories. Note that this might result in memory issues for
                algorithms that require lots of training data.
            save_test_trajs (bool): Save all test trajectories. Note that this might result in memory issues for
                algorithms that require lots of training data.
        """

        super().__init__(test_envs[0], ctrl, train_env=train_envs[0], safety_filter=None)

        self.train_envs = []
        self.test_envs = []
        if not(len(train_envs) == n_epochs) or len(train_envs) == 1:
            raise ValueError('The number of train_envs must match the number of epochs, or be a list of only a single env if only one is used during training.')
        for train_env in train_envs:
            env = None
            if not is_wrapped(train_env, RecordDataWrapper):
                env = RecordDataWrapper(train_env)
                self.train_envs.append(env)
        self.prior_ctrl_exists=False
        if hasattr(self.ctrl, 'prior_ctrl'):
            self.prior_ctrl_exists = True
        if (not(len(test_envs) == n_epochs+1) and self.prior_ctrl_exists) or \
                ( not(len(test_envs) == n_epochs) and not(self.prior_ctrl_exists) ) or len(test_envs) == 1:
            # We want one test environment per epoch such that all the testing is done with the same noise profile to
            # ensure consistency. Otherwise, only one test environment is required.
            raise ValueError('The number of test envs must match the number of epochs, or 1 plus the number of epochs if there is a prior_ctrl,'
                             ' or be a list of only a single env if only one is used for all evaluations.')
        for test_env in test_envs:
            env = None
            if not is_wrapped(env, RecordDataWrapper):
                env = RecordDataWrapper(test_env)
                self.test_envs.append(env)
        self.n_epochs = n_epochs
        self.n_train_episodes_per_epoch = n_train_episodes_per_epoch
        self.n_test_episodes_per_epoch = n_test_episodes_per_epoch
        self.output_dir = output_dir
        self.metrics = None
        self.save_train_trajs = save_train_trajs
        self.save_test_trajs = save_test_trajs
        self.all_train_data = None
        self.all_test_data = None

    def launch_training(self,
                        **kwargs):
        """Launches the epoch training.


        Returns:
            metrics (dict): Metric data from training. Then
        if self.save_train_trajs:
                Also returns:
                    all_train_data (dict): Dictionary of the training data keyed by the epoch
        if self.save_test_trajs:
                Also returns:
                    all_test_data (dict): Dictionary of the test data keyed by the epoch

        """
        epoch_start_ind = 0
        # Check if the controller has a prior_ctrl that should be run to collect data first.
        if hasattr(self.ctrl, 'prior_ctrl'):
            # Get prior model's performance
            eval_data = self._execute_task(ctrl=self.ctrl.prior_ctrl,
                                           env=self.test_envs[0],
                                           n_episodes=self.n_test_episodes_per_epoch,
                                           n_steps=None,
                                           log_freq=self.env.CTRL_FREQ)
            if self.save_test_trajs:
                self.add_to_all_test_data(eval_data)
            metrics = self.compute_metrics(eval_data)
            self.add_to_metrics(metrics)
            # Train with prior model.
            self.launch_single_train_epoch(self.ctrl.prior_ctrl,
                                           self.train_envs[0],
                                           self.n_train_episodes_per_epoch,
                                           0,
                                           None,
                                           self.env.CTRL_FREQ,
                                           **kwargs)
            eval_data = self._execute_task(ctrl=self.ctrl,
                                           env=self.test_envs[1],
                                           n_episodes=self.n_test_episodes_per_epoch,
                                           n_steps=None,
                                           log_freq=self.env.CTRL_FREQ)
            if self.save_test_trajs:
                self.add_to_all_test_data(eval_data)
            metrics = self.compute_metrics(eval_data)
            self.add_to_metrics(metrics)
            epoch_start_ind = 1
        for episode_i in range(epoch_start_ind, self.n_epochs):
            if len(self.train_envs) == 1:
                train_env_num = 0
            else:
                train_env_num = episode_i
            self.launch_single_train_epoch(self.ctrl,
                                           self.train_envs[train_env_num],
                                           self.n_train_episodes_per_epoch,
                                           episode_i,
                                           None,
                                           self.env.CTRL_FREQ,
                                           **kwargs)

            if len(self.test_envs) == 1:
                test_env_num = 0
            elif self.prior_ctrl_exists:
                test_env_num = episode_i + 1
            else:
                test_env_num = episode_i
            eval_data = self._execute_task(ctrl=self.ctrl,
                                           env=self.test_envs[test_env_num],
                                           n_episodes=self.n_test_episodes_per_epoch,
                                           n_steps=None,
                                           log_freq=self.env.CTRL_FREQ)
            if self.save_test_trajs:
                self.add_to_all_test_data(eval_data)
            metrics = self.compute_metrics(eval_data)
            self.add_to_metrics(metrics)

        # return learning stats
        return_values = [deepcopy(self.metrics)]
        if self.save_train_trajs:
            return_values.append(deepcopy(self.all_train_data))
        if self.save_test_trajs:
            return_values.append(deepcopy(self.all_test_data))
        return tuple(return_values)

    def launch_single_train_epoch(self,
                                  run_ctrl,
                                  env,
                                  n_episodes,
                                  episode_i,
                                  n_steps,
                                  log_freq,
                                  **kwargs):
        """Run a single training epoch

        Args:
            run_ctrl (BaseController): Controller used to collect the data.
            env (BenchmarkEnv): Environment to collect the data.
            n_episodes (int): Number of runs to execute.
            episode_i (int): The current episode.
            log_freq (int): The frequency with which to log information.

        Returns:

        """
        # Training Data Collection
        traj_data = self._execute_task(ctrl=run_ctrl,
                                       env=env,
                                       n_episodes=n_episodes,
                                       n_steps=n_steps,
                                       log_freq=log_freq)
        # Add trajectory data to all training data.
        if self.save_train_trajs:
            self.add_to_all_train_data(traj_data)
        # Parsing of training Data.
        train_inputs, train_outputs = self.preprocess_training_data(traj_data, **kwargs)
        # Learning of training data.
        self.train_controller(train_inputs, train_outputs, **kwargs)

    def preprocess_training_data(self, traj_data, **kwargs):
        """Preprocess the trajectory data into training data for controller training.

        Args:
            traj_data (dict): Dictionary of the trajectory data.

        """

        raise NotImplementedError('preprocessing_training_data not implemented!')

    def train_controller(self, train_inputs, train_outputs, **kwargs):
        """Train the controller from training data train_inputs and train_outputs. The intention is that
        preprocess_trianing_data has processed the data and formatted such that ctrl.learn should be able to learn
        from the given data.

        Args:
            train_inputs (np.array): Training data array.
            train_outputs (np.array): Training data targets.
        """
        train_output = self.ctrl.learn(input_data=train_inputs, target_data=train_outputs)
        raise NotImplementedError('Controller training needs to be implemented.')


    def add_to_all_train_data(self, traj_data):
        """Add data from single run to all training data.

        Args:
            traj_data (dict): Single training run data.

        """
        if self.all_train_data is None:
            self.all_train_data = deepcopy(traj_data)
        else:
            for key in self.all_train_data:
                self.all_train_data[key].append(deepcopy(traj_data[key][0]))

    def add_to_all_test_data(self, traj_data):
        """Add data from single run to all test data.

        Args:
            traj_data (dict): Single test run data.

        """
        if self.all_test_data is None:
            self.all_test_data = deepcopy(traj_data)
        else:
            for key in self.all_test_data:
                self.all_test_data[key].append(deepcopy(traj_data[key][0]))

    def add_to_metrics(self, metrics):
        """Collect metric data from a single run.

        Args:
            metrics (dict): Dictionary of run metrics.

        Returns:

        """
        if self.metrics is None:
            self.metrics = deepcopy(metrics)
        else:
            for key in self.metrics:
                self.metrics[key].append(deepcopy(metrics[key][0]))

    def compute_metrics(self, trajs_data):
        '''Compute all standard metrics on the given trajectory data.

        Args:
            trajs_data (defaultdict(list)): The raw data from the executed runs.

        Returns:
            metrics (dict): The metrics calculated from the raw data.
        '''

        metrics = super().compute_metrics(trajs_data)
        list_metrics = {}
        for key, value in metrics.items():
            list_metrics[key] = [value]
        return list_metrics
