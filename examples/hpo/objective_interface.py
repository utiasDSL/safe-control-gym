import os
import time
import sys
import yaml
import matplotlib.pyplot as plt
from filelock import FileLock
from functools import partial
from multiprocessing import Process, Manager
from copy import deepcopy
import numpy as np
import math

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config


def objective(config, result_metrics, result_paths, i):

    set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
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

    experiment = BaseExperiment(eval_env, control_agent)
    experiment.launch_training()
    data, metrics = experiment.run_evaluation(n_episodes=5, n_steps=None, done_on_max_steps=True)
    control_agent.close()

    # save to pickle
    with open(os.path.join(config.output_dir, 'metrics.pkl'), 'wb') as f:
        import pickle
        pickle.dump(metrics, f)

    result_metrics[i] = metrics
    result_paths[i] = config.output_dir
    avg_rmse = metrics['average_rmse']

    _, ax3 = plt.subplots()
    ax3.plot(data['obs'][0][:, 0], data['obs'][0][:, 2], 'r--', label='Agent Trajectory')
    ax3.plot(eval_env.X_GOAL[:, 0], eval_env.X_GOAL[:, 2], 'g--', label='Reference')
    ax3.scatter(data['obs'][0][0, 0], data['obs'][0][0, 2], color='g', marker='o', s=100, label='Initial State')
    ax3.set_xlabel(r'X')
    ax3.set_ylabel(r'Z')
    ax3.set_box_aspect(0.5)
    ax3.legend(loc='upper right')

    plt.tight_layout()
    # save the plot
    plt.savefig(os.path.join(config.output_dir, f'trajectory_rmse_{avg_rmse}.png'))

    return metrics


def main(config):

    # init protocol
    protocol = {'state': 'idle'}
    path = os.path.join(config.output_dir, config.tag)
    while protocol['state'] != 'end':
        try:
            # create a lock
            lock = FileLock(f'{path}/protocol.yaml.lock')
            with lock:
                with open(f'{path}/protocol.yaml', 'r')as f:
                    protocol = yaml.safe_load(f)    
        except:
            protocol = {'state': 'idle'}
        time.sleep(.5)

        if protocol:
            if protocol['state'] == 'request':
                hp_list = protocol['hps']
                repeat_eval = protocol['repeat_eval']
                max_processes = protocol['max_processes']
                metric = protocol['metric']
                output_dir = config.output_dir
                bo_algo = protocol['BO_algo']
                ALGO = protocol['algo']
                SYS = 'quadrotor_2D_attitude'

                if ALGO == 'gp_mpc':
                    PRIOR = '150'
                    sys.argv[1:] = ['--algo', ALGO,
                                    '--task', 'quadrotor',
                                    '--overrides',
                                        f'./examples/hpo/{ALGO}/config_overrides/{SYS}/{SYS}_track.yaml',
                                        f'./examples/hpo/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                                    '--output_dir', output_dir,
                                    '--tag', bo_algo,
                                    '--use_gpu', 'True'
                                    ]
                else:
                    PRIOR = ''
                    sys.argv[1:] = ['--algo', ALGO,
                                        '--task', 'quadrotor',
                                        '--overrides',
                                            f'./examples/hpo/rl/config_overrides/{SYS}/{SYS}_track.yaml',
                                            f'./examples/hpo/rl/{ALGO}/config_overrides/{SYS}/{ALGO}_{SYS}_{PRIOR}.yaml',
                                        '--output_dir', output_dir,
                                        '--tag', bo_algo,
                                        '--use_gpu', 'True'
                                        ]
                fac = ConfigFactory()
                config = fac.merge()

                aggregate_results = []
                aggregate_paths = []
                for hps in hp_list:
                    processes = []
                    result_metrics = Manager().list([None] * repeat_eval)
                    result_paths = Manager().list([None] * repeat_eval)
                    for i in range(repeat_eval):
                        if protocol['mode'] == 'train':
                            seed = np.random.randint(0, 10000)
                        else:
                            seed = np.random.randint(10000, 20000)
                        config.seed = seed
                        for hp in hps:
                            if hp == 'state_weight' or hp == 'state_dot_weight' or hp == 'action_weight':
                                if config.algo == 'gp_mpc':
                                    config.algo_config['q_mpc'] = [hps['state_weight'], hps['state_dot_weight'], hps['state_weight'], hps['state_dot_weight'], hps['state_weight']]
                                    config.algo_config['r_mpc'] = [hps['action_weight'], hps['action_weight']]
                                else:
                                    config.task_config['rew_state_weight'] = [hps['state_weight'], hps['state_dot_weight'], hps['state_weight'], hps['state_dot_weight'], hps['state_weight']]
                                    config.task_config['rew_action_weight'] = [hps['action_weight'], hps['action_weight']]
                            elif hp == 'actor_lr' or hp == 'critic_lr' or hp == 'entropy_lr' or hp == 'learning_rate':
                                config.algo_config[hp] = math.pow(10, hps[hp])
                            else:
                                config.algo_config[hp] = hps[hp]

                        p = Process(target=objective, args=(deepcopy(config), result_metrics, result_paths, i))
                        processes.append(p)
                
                    step = 0
                    while step < len(processes):
                        begin = int(step * max_processes)
                        end = min(begin+max_processes, len(processes))
                        for p in processes[begin:end]:
                            p.start()
                        for p in processes[begin:end]:
                            p.join()
                        step += 1

                    # Collect results from all processes
                    results = []
                    paths = []
                    for i in range(repeat_eval):
                        results.append(float(result_metrics[i][metric]))
                        paths.append(result_paths[i])
                    aggregate_results.append(results)
                    aggregate_paths.append(paths)
                    
                protocol['y'] = aggregate_results
                protocol['saved_paths'] = aggregate_paths
                
                protocol['state'] = 'done'
                # create a lock
                lock = FileLock(f'{path}/protocol.yaml.lock')
                with lock:
                    with open(f'{path}/protocol.yaml', 'w')as f:
                        yaml.dump(protocol, f, default_flow_style=False, sort_keys=False)
                

                # with open(f'{path}/tmp.yaml', 'w')as f:
                #     yaml.dump(protocol, f, default_flow_style=False, sort_keys=False)
                # os.rename(f'{path}/tmp.yaml', f'{path}/protocol.yaml')
            
            elif protocol['state'] == 'end':
                break
        else:
            protocol = {'state': 'idle'}

    return


if __name__ == '__main__':
    # Make config.
    fac = ConfigFactory()
    # merge config
    config = fac.merge()

    main(config)