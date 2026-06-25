'''Run an existing safe-control-gym controller through a networked wrapper.'''

from functools import partial
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe_control_gym.controllers.networked import DynamicPETCTrigger, NetworkedControllerWrapper, uniform_delay
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=False, n_episodes=1, n_steps=None, max_delay_steps=3, sample_period_steps=1, seed=0):
    '''Run the configured controller with event-triggered delayed observations.

    The configured controller can be LQR, iLQR, MPC, or another controller with
    the standard safe-control-gym select_action(obs, info) method.
    '''

    config = ConfigFactory().merge()
    env_func = partial(make, config.task, **config.task_config)
    random_env = env_func(gui=False)

    base_ctrl = make(config.algo, env_func, **config.algo_config)
    rng = np.random.default_rng(seed)

    all_network_metrics = []
    n_episodes = 1 if n_episodes is None else n_episodes

    for _ in range(n_episodes):
        init_obs, _ = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_obs)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_obs)

        obs_dim = int(np.asarray(init_obs).reshape(-1).size)
        ctrl = NetworkedControllerWrapper(
            base_ctrl,
            observation_dim=obs_dim,
            sample_period_steps=sample_period_steps,
            max_delay_steps=max_delay_steps,
            delay_sampler=uniform_delay(max_delay_steps, rng=rng),
            trigger=DynamicPETCTrigger(
                error_weight=1.0,
                signal_weight=0.03,
                eta_decay=0.02,
                eta_growth=0.02,
                eta_skip_gain=0.15,
                eta_transmit_drop=0.3,
                eta_max=0.5,
            ),
        )
        ctrl.reset_before_run(obs=init_obs, env=static_env)

        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
        experiment.launch_training()
        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        metrics = experiment.compute_metrics(trajs_data)
        network_metrics = ctrl.metrics.summary()
        all_network_metrics.append(network_metrics)

        print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))
        print('NETWORK METRICS - ' + ', '.join([f'{key}: {value}' for key, value in network_metrics.items()]))

        static_env.close()
        static_train_env.close()

    base_ctrl.close()
    random_env.close()
    return all_network_metrics


if __name__ == '__main__':
    run()
