import os

import munch
import yaml
import numpy as np
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_seed_from_config, set_device_from_config
from safe_control_gym.hyperparameters.hpo import HPO

# To set relative pathing of experiment imports.
import sys
import os.path as path
from experiments.comparisons.gpmpc.gpmpc_plotting_utils import make_plots, gather_training_samples, plot_data_eff_from_csv, plot_hpo_eval

def hpo(config):
    """Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.

    """

    if config.hpo_config.hpo:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # HPO
    hpo = HPO(config.algo, 
              config.task, 
              config.sampler,
              config.load_study, 
              config.output_dir, 
              config.task_config, 
              config.hpo_config, 
              **config.algo_config)
    
    if config.hpo_config.hpo:
        hpo.hyperparameter_optimization()
        print("Hyperparameter optimization done.")
    elif config.hpo_config.perturb_hps and config.opt_hps != "":
        hpo._perturb_hps(hp_path = config.opt_hps)
        print("Perturbed hyperparameter files have been produced.")
    else:
        raise ValueError("hpo or perturb_hps must be set to True.")  

def train(config):
    """Training.

    Usage:
        * to start training, use with `--func train`.

    """

    # Override algo_config with given yaml file
    if config.opt_hps == "":
        # if no opt_hps file is given
        pass
    else:
        # if opt_hps file is given
        with open(config.opt_hps, "r") as f:
            opt_hps = yaml.load(f, Loader=yaml.FullLoader)
        for hp in opt_hps:
            if isinstance(config.algo_config[hp], list) and not isinstance(opt_hps[hp], list):
                config.algo_config[hp] = [opt_hps[hp]] * len(config.algo_config[hp])
            else:
                config.algo_config[hp] = opt_hps[hp]

    set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    config.algo_config.output_dir = config.output_dir
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                use_gpu=config.use_gpu,
                **config.algo_config
                )
    ctrl.reset()

    # Note that the following script is the same as train_runs, test_runs = ctrl._learn()
    # but with some plotting along the way.

    num_epochs = config.algo_config.num_epochs
    num_train_episodes_per_epoch = config.algo_config.num_train_episodes_per_epoch
    num_test_episodes_per_epoch = config.algo_config.num_test_episodes_per_epoch
    num_samples = config.algo_config.num_samples
    train_runs = {0: {}}
    test_runs = {0: {}}

    if config.algo_config.same_train_initial_state:
        train_envs = []
        for epoch in range(num_epochs):
            train_envs.append(env_func(randomized_init=True, seed=config.seed))
            train_envs[epoch].action_space.seed(config.seed)
    else:
        train_env = env_func(randomized_init=True, seed=config.seed)
        train_env.action_space.seed(config.seed)
        train_envs = [train_env]*num_epochs
    #init_test_states = get_random_init_states(env_func, num_test_episodes_per_epoch)
    test_envs = []
    if config.algo_config.same_test_initial_state:
        for epoch in range(num_epochs):
            test_envs.append(env_func(randomized_init=True, seed=config.seed*222))
            test_envs[epoch].action_space.seed(config.seed*222)
    else:
        test_env = env_func(randomized_init=True, seed=config.seed*222)
        test_env.action_space.seed(config.seed*222)
        test_envs = [test_env]*num_epochs


    for episode in range(num_train_episodes_per_epoch):
        run_results = ctrl.prior_ctrl.run(env=train_envs[0],
                                          terminate_run_on_done=config.algo_config.terminate_train_on_done)
        train_runs[0].update({episode: munch.munchify(run_results)})
        ctrl.reset()
    for test_ep in range(num_test_episodes_per_epoch):
        run_results = ctrl.run(env=test_envs[0],
                               terminate_run_on_done=config.algo_config.terminate_test_on_done)
        test_runs[0].update({test_ep: munch.munchify(run_results)})
    ctrl.reset()

    for epoch in range(1, num_epochs):
        # only take data from the last episode from the last epoch
        if config.algo_config.rand_data_selection:
            x_seq, actions, x_next_seq = gather_training_samples(train_runs, epoch-1, num_samples, train_envs[epoch-1].np_random)
        else:
            x_seq, actions, x_next_seq = gather_training_samples(train_runs, epoch-1, num_samples)
        train_inputs, train_outputs = ctrl.preprocess_training_data(x_seq, actions, x_next_seq)
        _ = ctrl.learn(input_data=train_inputs, target_data=train_outputs)

        # Test new policy.
        test_runs[epoch] = {}
        for test_ep in range(num_test_episodes_per_epoch):
            ctrl.x_prev = test_runs[epoch-1][episode]['obs'][:ctrl.T+1,:].T
            ctrl.u_prev = test_runs[epoch-1][episode]['action'][:ctrl.T,:].T
            ctrl.reset()
            run_results = ctrl.run(env=test_envs[epoch],
                                   terminate_run_on_done=config.algo_config.terminate_test_on_done)
            test_runs[epoch].update({test_ep: munch.munchify(run_results)})
        # gather training data
        train_runs[epoch] = {}
        for episode in range(num_train_episodes_per_epoch):
            ctrl.reset()
            ctrl.x_prev = train_runs[epoch-1][episode]['obs'][:ctrl.T+1,:].T
            ctrl.u_prev = train_runs[epoch-1][episode]['action'][:ctrl.T,:].T
            run_results = ctrl.run(env=train_envs[epoch],
                                   terminate_run_on_done=config.algo_config.terminate_train_on_done)
            train_runs[epoch].update({episode: munch.munchify(run_results)})

        lengthscale, outputscale, noise, kern = ctrl.gaussian_process.get_hyperparameters(as_numpy=True)

        trajectory = 0
        np.savez(os.path.join(config.output_dir, 'data_%s' % epoch),
                 train_runs=train_runs,
                 test_runs=test_runs,
                 num_epochs=num_epochs,
                 num_train_episodes_per_epoch=num_train_episodes_per_epoch,
                 num_test_episodes_per_epoch=num_test_episodes_per_epoch,
                 num_samples=num_samples,
                 trajectory=trajectory,
                 ctrl_freq=config.task_config.ctrl_freq,
                 lengthscales=lengthscale,
                 outputscale=outputscale,
                 noise=noise,
                 kern=kern,
                 train_data=ctrl.train_data,
                 test_data=ctrl.test_data,
                 data_inputs=ctrl.data_inputs,
                 data_targets=ctrl.data_targets)

        make_plots(test_runs, train_runs, train_envs[0].state_dim, config.output_dir)

    fname = os.path.join(config.output_dir, 'figs', 'avg_rmse_cost_learning_curve.csv')
    plot_data_eff_from_csv(fname, 'Cartpole Data Efficiency')
    # plot_runs(test_runs, num_epochs)
    return train_runs, test_runs

def plot(config):

    plot_hpo_eval(config)


MAIN_FUNCS = {"hpo": hpo, "train": train, "plot": plot}

if __name__ == "__main__":

    fac = ConfigFactory()
    fac.add_argument("--plot_dir", type=str, default='', help="Create plot from CSV file.")
    fac.add_argument("--opt_hps", type=str, default="", help="yaml file as a result of HPO.")
    fac.add_argument("--func", type=str, default="train", help="main function to run.")
    fac.add_argument("--load_study", type=bool, default=False, help="whether to load study from a previous HPO.")
    fac.add_argument("--sampler", type=str, default="TPESampler", help="which sampler to use in HPO.")
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))
    func(config)
