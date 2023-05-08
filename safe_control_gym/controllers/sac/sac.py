'''Soft Actor Critic (SAC)

Adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py

References papers & code:
    * [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)
    * [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)
    * [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/pdf/1910.07207.pdf)
    * [openai spinning up - sac](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac)
    * [rlkit - sac](https://github.com/vitchyr/rlkit/tree/7daf34b0ef2277d545a0ee792399a2ae6c3fb6ad/rlkit/torch/sac)
    * [ray rllib - sac](https://github.com/ray-project/ray/tree/master/rllib/agents/sac)
    * [curl - curl_sac](https://github.com/MishaLaskin/curl/blob/master/curl_sac.py)
'''

import os
import time
from collections import defaultdict

import torch
import numpy as np

from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, set_random_state, is_wrapped
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_obs, _unflatten_obs
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics
from safe_control_gym.math_and_models.normalization import BaseNormalizer, MeanStdNormalizer, RewardStdNormalizer

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.sac.sac_utils import SACAgent, SACBuffer


class SAC(BaseController):
    '''soft actor critic.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)

        # task
        if self.training:
            # training (+ evaluation)
            self.env = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.env = VecRecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
        else:
            # testing only
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)

        # agent
        self.agent = SACAgent(self.env.observation_space,
                              self.env.action_space,
                              hidden_dim=self.hidden_dim,
                              gamma=self.gamma,
                              tau=self.tau,
                              init_temperature=self.init_temperature,
                              use_entropy_tuning=self.use_entropy_tuning,
                              target_entropy=self.target_entropy,
                              actor_lr=self.actor_lr,
                              critic_lr=self.critic_lr,
                              entropy_lr=self.entropy_lr)
        self.agent.to(self.device)

        # pre-/post-processing
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(shape=self.env.observation_space.shape, clip=self.clip_obs, epsilon=1e-8)

        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8)

        # logging
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # disable logging to texts and tfboard for testing
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        '''Prepares for training or testing.'''
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, _ = self.env.reset()
            self.obs = self.obs_normalizer(obs)
            self.buffer = SACBuffer(self.env.observation_space, self.env.action_space, self.max_buffer_size, self.train_batch_size)
        else:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self, path, save_buffer=True):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)

        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict()
        }
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'random_state': get_random_state(),
                'env_random_state': self.env.get_env_random_state()
            }
            # latest checkpoint shoud enable save_buffer (for experiment restore),
            # but intermediate checkpoint shoud not, to save storage (buffer is large)
            if save_buffer:
                exp_state['buffer'] = self.buffer.state_dict()
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self, path):
        '''Restores model and experiment given checkpoint path.'''
        state = torch.load(path)

        # restore params
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])

        # restore experiment state
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            set_random_state(state['random_state'])
            self.env.set_env_random_state(state['env_random_state'])
            if 'buffer' in state:
                self.buffer.load_state_dict(state['buffer'])
            self.logger.load(self.total_steps)

    def learn(self, env=None, **kwargs):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        while self.total_steps < self.max_env_steps:
            results = self.train_step()

            # checkpoint
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # latest/final checkpoint
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
            if self.num_checkpoints and self.total_steps % (self.max_env_steps // self.num_checkpoints) == 0:
                # intermediate checkpoint
                path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                self.save(path, save_buffer=False)

            # eval
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(eval_results['ep_lengths'].mean(),
                                                                                                            eval_results['ep_lengths'].std(),
                                                                                                            eval_results['ep_returns'].mean(),
                                                                                                            eval_results['ep_returns'].std()))
                # save best model
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))

            # logging
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = self.agent.ac.act(obs, deterministic=True)

        return action

    def run(self, env=None, render=False, n_episodes=10, verbose=False, **kwargs):
        '''Runs evaluation with current policy.'''
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, info = env.reset()
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []

        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)

            obs, _, done, info = env.step(action)
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            if verbose:
                print(f'obs {obs} | act {action}')

            if done:
                assert 'episode' in info
                ep_returns.append(info['episode']['r'])
                ep_lengths.append(info['episode']['l'])
                obs, info = env.reset()
            obs = self.obs_normalizer(obs)

        # collect evaluation results
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths}
        if len(frames) > 0:
            eval_results['frames'] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def train_step(self, **kwargs):
        '''Performs a training step.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        obs = self.obs
        start = time.time()

        if self.total_steps < self.warm_up_steps:
            act = np.stack([self.env.action_space.sample() for _ in range(self.rollout_batch_size)])
        else:
            with torch.no_grad():
                act = self.agent.ac.act(torch.FloatTensor(obs).to(self.device), deterministic=False)
        next_obs, rew, done, info = self.env.step(act)

        next_obs = self.obs_normalizer(next_obs)
        rew = self.reward_normalizer(rew, done)
        mask = 1 - np.asarray(done)

        # time truncation is not true termination
        terminal_idx, terminal_obs = [], []
        for idx, inf in enumerate(info['n']):
            if 'terminal_info' not in inf:
                continue
            inff = inf['terminal_info']
            if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                terminal_idx.append(idx)
                terminal_obs.append(inf['terminal_observation'])
        if len(terminal_obs) > 0:
            terminal_obs = _unflatten_obs(self.obs_normalizer(_flatten_obs(terminal_obs)))

        # collect the true next states and masks (accounting for time truncation)
        true_next_obs = _unflatten_obs(next_obs)
        true_mask = mask.copy()
        for idx, term_ob in zip(terminal_idx, terminal_obs):
            true_next_obs[idx] = term_ob
            true_mask[idx] = 1.0
        true_next_obs = _flatten_obs(true_next_obs)

        self.buffer.push({
            'obs': obs,
            'act': act,
            'rew': rew,
            # 'next_obs': next_obs,
            # 'mask': mask,
            'next_obs': true_next_obs,
            'mask': true_mask,
        })
        obs = next_obs

        self.obs = obs
        self.total_steps += self.rollout_batch_size

        # learn
        results = defaultdict(list)
        if self.total_steps > self.warm_up_steps and not self.total_steps % self.train_interval:
            # Regardless of how long you wait between updates,
            # the ratio of env steps to gradient steps is locked to 1.
            # alternatively, can update once each step
            for _ in range(self.train_interval):
                batch = self.buffer.sample(self.train_batch_size, self.device)
                res = self.agent.update(batch)
                for k, v in res.items():
                    results[k].append(v)

        results = {k: sum(v) / len(v) for k, v in results.items()}
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def log_step(self, results):
        '''Does logging after a training step.'''
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'time': results['elapsed_time'],
                'progress': step / self.max_env_steps,
            },
            step,
            prefix='time',
            write=False,
            write_tb=False)

        # learning stats
        if 'policy_loss' in results:
            self.logger.add_scalars(
                {
                    k: results[k]
                    for k in ['policy_loss', 'critic_loss', 'entropy_loss']
                },
                step,
                prefix='loss')

        # performance stats
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        ep_constraint_violation = np.asarray(self.env.queued_stats['constraint_violation'])
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_reward': (ep_returns / ep_lengths).mean(),
                'ep_constraint_violation': ep_constraint_violation.mean()
            },
            step,
            prefix='stat')

        # total constraint violation during learning
        total_violations = self.env.accumulated_stats['constraint_violation']
        self.logger.add_scalars({'constraint_violation': total_violations}, step, prefix='stat')

        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            eval_constraint_violation = results['eval']['constraint_violation']
            eval_mse = results['eval']['mse']
            self.logger.add_scalars(
                {
                    'ep_length': eval_ep_lengths.mean(),
                    'ep_return': eval_ep_returns.mean(),
                    'ep_reward': (eval_ep_returns / eval_ep_lengths).mean(),
                    'constraint_violation': eval_constraint_violation.mean(),
                    'mse': eval_mse.mean()
                },
                step,
                prefix='stat_eval')

        # print summary table
        self.logger.dump_scalars()
