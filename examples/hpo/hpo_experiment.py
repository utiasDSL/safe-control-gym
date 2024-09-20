"""Template hyperparameter optimization/hyperparameter evaluation script.

"""
import os
from functools import partial

import yaml
import wandb
import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Environment, Task

from safe_control_gym.hyperparameters.hpo import HPO
from safe_control_gym.hyperparameters.hpo_vizier import HPO as HPO_vizier
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config, mkdirs


def hpo(config):
    """Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.

    """

    # change the cost function for rl methods
    if config.algo == 'ppo':
        config.task_config.cost = 'rl_reward'
        config.task_config.obs_goal_horizon = 1
        config.normalized_rl_action_space = True
        config.task_config.disturbances.observation[0]['std'] += [0, 0, 0, 0, 0, 0]
        config.algo_config.log_interval = 10000000
        config.algo_config.eval_interval = 10000000
    elif config.algo == 'gp_mpc' or config.algo == 'gpmpc_acados' or config.algo == 'ilqr':
        pass
    else:
        raise ValueError('Only ppo, gp_mpc, gpmpc_acados, and ilqr are supported for now.')
    
    # Experiment setup.
    if config.hpo_config.hpo:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # initialize safety filter
    if 'safety_filter' not in config:
        config.safety_filter = None
        config.sf_config = None

    wandb.init(project='hyperparameter-optimization', config=config, group='HPO')

    # HPO
    if config.sampler == 'Optuna':
        hpo = HPO(config.algo,
                config.task,
                config.load_study,
                config.output_dir,
                config.task_config,
                config.hpo_config,
                config.algo_config,
                config.safety_filter,
                config.sf_config,
                )
    elif config.sampler == 'Vizier':
        hpo = HPO_vizier(config.algo,
                        config.task,
                        config.load_study,
                        config.output_dir,
                        config.task_config,
                        config.hpo_config,
                        config.algo_config,
                        config.safety_filter,
                        config.sf_config,
                        )
    else:
        raise ValueError('Only Optuna and Vizier are supported for now.')


    if config.hpo_config.hpo:
        hpo.hyperparameter_optimization()
        print('Hyperparameter optimization done.')


def train(config):
    """Training for a given set of hyperparameters.

    Usage:
        * to start training, use with `--func train`.

    """
    # change the cost function for rl methods
    if config.algo == 'ppo':
        config.task_config.cost = 'rl_reward'
        config.task_config.obs_goal_horizon = 1
        config.normalized_rl_action_space = True
        config.task_config.disturbances.observation[0]['std'] += [0, 0, 0, 0, 0, 0]
    elif config.algo == 'gp_mpc' or config.algo == 'gpmpc_acados' or config.algo == 'ilqr':
        pass
    else:
        raise ValueError('Only ppo, gp_mpc, gpmpc_acados, and ilqr are supported for now.')
    # Override algo_config with given yaml file
    if config.opt_hps == '':
        # if no opt_hps file is given
        pass
    else:
        # if opt_hps file is given
        with open(config.opt_hps, 'r') as f:
            opt_hps = yaml.load(f, Loader=yaml.FullLoader)
        for hp in opt_hps:
            if hp == 'state_weight' or hp == 'state_dot_weight' or hp == 'action_weight':
                    if config.algo == 'gp_mpc' or config.algo == 'gpmpc_acados':
                        if config.task == 'cartpole':
                            config.algo_config['q_mpc'] = [opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight']]
                            config.algo_config['r_mpc'] = [opt_hps['action_weight']]
                        elif config.task == 'quadrotor':
                            config.algo_config['q_mpc'] = [opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight']]
                            config.algo_config['r_mpc'] = [opt_hps['action_weight'], opt_hps['action_weight']]
                        else:
                            raise ValueError('Only cartpole and quadrotor tasks are supported for gp_mpc.')
                    elif config.algo == 'ilqr':
                        if config.task == 'cartpole':
                            config.algo_config['q_lqr'] = [opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight']]
                            config.algo_config['r_lqr'] = [opt_hps['action_weight']]
                        elif config.task == 'quadrotor':
                            #TODO if implemented for quadrotor, pitch rate penalty should be small.
                            # raise ValueError('Only cartpole task is supported for ilqr.')
                            config.algo_config['q_lqr'] = [opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight']]
                            config.algo_config['r_lqr'] = [opt_hps['action_weight'], opt_hps['action_weight']]
                        else:
                            raise ValueError('Only cartpole and quadrotor tasks are supported for ilqr.')
                    else:
                        if config.task == 'cartpole':
                            config.task_config['rew_state_weight'] = [opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight']]
                            config.task_config['rew_action_weight'] = [opt_hps['action_weight']]
                        elif config.task == 'quadrotor':
                            config.task_config['rew_state_weight'] = [opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight'], opt_hps['state_weight'], opt_hps['state_dot_weight']]
                            config.task_config['rew_action_weight'] = [opt_hps['action_weight'], opt_hps['action_weight']]
            elif isinstance(config.algo_config[hp], list) and not isinstance(opt_hps[hp], list):
                config.algo_config[hp] = [opt_hps[hp]] * len(config.algo_config[hp])
            else:
                config.algo_config[hp] = opt_hps[hp]
    # Experiment setup.
    if config.plot_best is False:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    # Note:
    # eval_env will take config.seed * 111 as its seed
    # env will take config.seed as its seed
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()

    eval_env = env_func(seed=config.seed * 111)

    # Setup safety filter
    # check if config has key safety_filter
    if 'safety_filter' in config:
        if config.safety_filter != '':
            env_func_filter = partial(make,
                                    config.task,
                                    seed=config.seed,
                                    **config.task_config)
            safety_filter = make(config.safety_filter,
                                env_func_filter,
                                seed=config.seed,
                                **config.sf_config)
            safety_filter.reset()

    if config.plot_best:
        ## rl
        control_agent.load('/home/tueilsy-st01/safe-control-gym/examples/hpo/results/2D_attitude/seed6_May-31-21-55-42_v0.5.0-623-g8229ba3/model_latest.pt')
        # control_agent.load('/home/tueilsy-st01/safe-control-gym/examples/hpo/rl/ppo/model_latest_ppo.pt')

        ## gpmpc
        #control_agent.load('/home/tueilsy-st01/safe-control-gym/examples/hpo/results/temp/seed2_May-29-11-31-10_v0.5.0-619-gdbfb011')
        experiment = BaseExperiment(eval_env, control_agent)
    else:
        if 'safety_filter' in config:
            if config.safety_filter != '':
                safety_filter.learn()
                mkdirs(f'{config.output_dir}/models/')
                safety_filter.save(path=f'{config.output_dir}/models/{config.safety_filter}.pkl')
                experiment = BaseExperiment(eval_env, control_agent, safety_filter=safety_filter)
        else:
            experiment = BaseExperiment(eval_env, control_agent)
            experiment.launch_training()
    results, metrics = experiment.run_evaluation(n_episodes=config.n_episodes, n_steps=None, done_on_max_steps=True)
    control_agent.close()

    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
        if config.task_config.quad_type == 4:
            system = 'quadrotor_2D'
    else:
        system = config.task

    if config.task == Environment.QUADROTOR:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4

        # plot results['current_physical_action']
        _, ax_act = plt.subplots()
        ax_act.plot(results['current_physical_action'][0][:, 0], 'b', label='Thrust')
        if config.task_config.quad_type == 4:
            ax_act.plot(results['current_physical_action'][0][:, 1], 'r', label='Pitch')
        else:
            ax_act.plot(results['current_physical_action'][0][:, 1], 'r', label='Thrust')
        ax_act.legend()
        ax_act.set_xlabel('Step')
        ax_act.set_ylabel('Input')
        plt.savefig(os.path.join(config.output_dir, 'inputs.png'))
        
        if config.task_config.quad_type != 4:
            _, ax = plt.subplots()
            ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='Agent Trajectory')
            ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
            ax.set_xlabel(r'$\theta$')
            ax.set_ylabel(r'$\dot{\theta}$')
            ax.set_box_aspect(0.5)
            ax.legend(loc='upper right')
            # save the plot
            plt.savefig(os.path.join(config.output_dir, 'trajectory_theta_theta_dot.png'))

            if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
                _, ax2 = plt.subplots()
                ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:, 0], 'r--', label='Agent Trajectory')
                ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), eval_env.X_GOAL[:, 0], 'b', label='Reference')
                ax2.set_xlabel(r'Time')
                ax2.set_ylabel(r'X')
                ax2.set_box_aspect(0.5)
                ax2.legend(loc='upper right')
                # save the plot
                plt.savefig(os.path.join(config.output_dir, 'trajectory_time_x.png'))
            elif config.task == Environment.QUADROTOR:
                _, ax2 = plt.subplots()
                ax2.plot(results['obs'][0][:, graph3_1 + 1], results['obs'][0][:, graph3_2 + 1], 'r--', label='Agent Trajectory')
                ax2.set_xlabel(r'x_dot')
                ax2.set_ylabel(r'z_dot')
                ax2.set_box_aspect(0.5)
                ax2.legend(loc='upper right')
                # save the plot
                plt.savefig(os.path.join(config.output_dir, 'trajectory_x_dot_z_dot.png'))

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='Agent Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(eval_env.X_GOAL[:, graph3_1], eval_env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        # save the plot
        plt.savefig(os.path.join(config.output_dir, 'trajectory_x.png'))

    # save to pickle
    with open(os.path.join(config.output_dir, 'metrics.pkl'), 'wb') as f:
        import pickle
        pickle.dump(metrics, f)
    with open(os.path.join(config.output_dir, 'results.pkl'), 'wb') as f:
        import pickle
        pickle.dump(results, f)
    
    return eval_env.X_GOAL, results, metrics


MAIN_FUNCS = {'hpo': hpo, 'train': train}


if __name__ == '__main__':

    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='Optuna', help='which sampler to use in HPO.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    fac.add_argument('--plot_best', type=bool, default=False, help='plot best agent trajectory.')
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception('Main function {} not supported.'.format(config.func))
    func(config)
