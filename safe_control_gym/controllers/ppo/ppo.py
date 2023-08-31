'''Proximal Policy Optimization (PPO)

Based on:
    * https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    * (hyperparameters) https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

Additional references:
    * Proximal Policy Optimization Algorithms - https://arxiv.org/pdf/1707.06347.pdf
    * Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO - https://arxiv.org/pdf/2005.12729.pdf
    * pytorch-a2c-ppo-acktr-gail - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    * openai spinning up - ppo - https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
    * stable baselines3 - ppo - https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo
    * Jump Start RL - https://arxiv.org/pdf/2204.02372.pdf& 
'''

import os
import time
import numpy as np
import torch
from functools import partial
from copy import deepcopy

from safe_control_gym.controllers.mpc.linear_mpc import LinearMPC
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, set_random_state, is_wrapped
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics
from safe_control_gym.math_and_models.normalization import BaseNormalizer, MeanStdNormalizer, RewardStdNormalizer
from safe_control_gym.envs.benchmark_env import Task

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.ppo.ppo_utils import PPOAgent, PPOBuffer, compute_returns_and_advantages


class PPO(BaseController):
    '''Proximal policy optimization.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 prior_info: dict=None,
                 encode_prior: bool=False,
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        
        self.encode_prior = encode_prior
        if encode_prior:
            assert prior_info is not None, ValueError("Please define prior info")

            # encode prior knowledge
            self.prior_info = prior_info
            self.prior_info['prior_prop'].update((prop, val * self.prior_info.prior_param_coeff) for prop, val in prior_info['prior_prop'].items())
            self.prior_env_func = partial(env_func, 
                                          inertial_prop=self.prior_info['prior_prop'], 
                                          info_in_reset=False, cost='quadratic',
                                          normalized_rl_action_space=False)

            # Initialize the method using linear MPC.
            self.prior_ctrl = LinearMPC(
                self.prior_env_func,
                horizon=self.prior_info.horizon,
                q_mpc=self.prior_info.q_mpc,
                r_mpc=self.prior_info.r_mpc,
                warmstart=self.prior_info.warmstart,
                soft_constraints=self.prior_info.soft_constraints['prior_soft_constraints'],
                terminate_run_on_done=True,
                prior_info=self.prior_info,
            )
            self.prior_ctrl.reset()

        # Task.
        if self.training:
            # Training and testing.
            self.env = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.env = VecRecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
            if encode_prior:
                prior_eval_env_func = partial(env_func, info_in_reset=False, cost='quadratic', normalized_rl_action_space=False)
                self.prior_eval_env = prior_eval_env_func(seed=seed * 111)
                # this is only used for combined policy evaluation
                self._prior_eval_env = env_func(seed=seed * 111)
        else:
            # Testing only.
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)
        # Agent.
        self.agent = PPOAgent(self.env.observation_space,
                              self.env.action_space,
                              hidden_dim=self.hidden_dim,
                              use_clipped_value=self.use_clipped_value,
                              clip_param=self.clip_param,
                              target_kl=self.target_kl,
                              entropy_coef=self.entropy_coef,
                              actor_lr=self.actor_lr,
                              critic_lr=self.critic_lr,
                              opt_epochs=self.opt_epochs,
                              mini_batch_size=self.mini_batch_size,
                              activation=self.activation)
        self.agent.to(self.device)
        # Pre-/post-processing.
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(shape=self.env.observation_space.shape, clip=self.clip_obs, epsilon=1e-8)
        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8)
        # Logging.
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # Disable logging to file and tfboard for evaluation.
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        '''Do initializations for training or evaluation.'''
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, _ = self.env.reset()
            self.obs = self.obs_normalizer(obs)
            if self.encode_prior:
                self.prior_eval_env.reset()
                self._prior_eval_env.reset()
        else:
            # Add episodic stats to be tracked.
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')
        if self.encode_prior:
            self.prior_ctrl.reset()

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.eval_env.close()
            if self.encode_prior:
                self.prior_eval_env.close()
                self._prior_eval_env.close()
        self.logger.close()

    def save(self,
             path
             ):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
        }
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'random_state': get_random_state(),
                'env_random_state': self.env.get_env_random_state()
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self,
             path
             ):
        '''Restores model and experiment given checkpoint path.'''
        state = torch.load(path)
        # Restore policy.
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        # Restore experiment state.
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            set_random_state(state['random_state'])
            self.env.set_env_random_state(state['env_random_state'])
            self.logger.load(self.total_steps)

    def learn_with_prior(self,
                    env=None,
                    **kwargs
                    ):
        """encode policy prior using the method inspired by jump start reinforcement learning 
           https://arxiv.org/pdf/2204.02372.pdf&
        
        """
        # check guide policy exist
        assert self.prior_ctrl is not None, ValueError("Please define guide policy")

        guide_step = np.arange(0, self.rollout_steps, self.discount_steps, dtype=int)
        # arrage array value from high to low
        guide_step = guide_step[::-1]

        def _is_moving_avg_increase(arr):
            """check if the moving average of array is increasing
            """
            if len(arr) == 0:
                return False
            window_size = 5
            if len(arr) < window_size:
                moving_avg = np.correlate(arr, np.ones(len(arr))) / len(arr)
            else:
                moving_avg = np.correlate(arr, np.ones(window_size)) / window_size

            if len(arr) > window_size * 3:
                return moving_avg[-1] > moving_avg[-2 * window_size] * self.improving_factor
            else:
                return False
        step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
        interval_save = np.zeros_like(step_interval, dtype=bool)
        old_total_steps = 0
        for h in guide_step:
            # it may not converge before the max_env_steps
            if self.total_steps > self.max_env_steps:
                break
            explore_policy_rewards = []
            enough = False
            increment = (self.rollout_steps- h) * self.rollout_batch_size
            steps_in_guidance = 0
            if h == 0: # it means explore policy can take over completely
                # compensate some steps such that the total steps is the same as the last step
                will_be_done = self.rollout_batch_size * self.rollout_steps * int((self.max_env_steps - self.total_steps)/(self.rollout_batch_size * self.rollout_steps))
                rest_steps = self.max_env_steps - self.total_steps - will_be_done
                if rest_steps == 0:
                    break
                horizon = int(rest_steps/self.rollout_batch_size)
                self._train_step(guide=True, h=0, pure_guide_horizon=horizon)
                break
            # it may not converge before the max_env_steps
            if self.total_steps > self.max_env_steps:
                break
            while _is_moving_avg_increase(explore_policy_rewards) == False and enough == False:
                # it may not converge before the max_env_steps
                if self.total_steps > self.max_env_steps:
                    break
                results, explore_reward = self._train_step(guide=True, h=h, pure_guide_horizon=self.rollout_steps)
                explore_policy_rewards.append(explore_reward)
                steps_in_guidance = self.total_steps - old_total_steps
                # Evaluation.
                if int(steps_in_guidance / increment) % 1 == 0 and self.total_steps > self.breaking_steps or int(steps_in_guidance / increment) % self.eval_every == 0:
                    _,  explore_policy_reward, expected_explore_policy_reward, _ = self._eval_combined_policy(h=h)
                    if explore_policy_reward > expected_explore_policy_reward:
                        enough = True
                interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
                if interval_save[interval_id] == False:
                    # Intermediate checkpoint.
                    path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
                    self.save(path)
                    interval_save[interval_id] = True
            old_total_steps = self.total_steps

        # finish the rest of the training
        while self.total_steps < self.max_env_steps:
            results = self.train_step()
            interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
            if interval_save[interval_id] == False:
                # Intermediate checkpoint.
                path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
                self.save(path)
                interval_save[interval_id] = True
        
        # Final checkpoint.
        path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
        self.save(path)

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        if self.encode_prior:
            self.learn_with_prior(env=env, **kwargs)
            return
        
        if self.num_checkpoints > 0:
            step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
            interval_save = np.zeros_like(step_interval, dtype=bool)
        while self.total_steps < self.max_env_steps:
            results = self.train_step()
            # Checkpoint.
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
                path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
                self.save(path)
            if self.num_checkpoints > 0:
                interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
                if interval_save[interval_id] == False:
                    # Intermediate checkpoint.
                    path = os.path.join(self.output_dir, "checkpoints", f'model_{self.total_steps}.pt')
                    self.save(path)
                    interval_save[interval_id] = True
            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(eval_results['ep_lengths'].mean(),
                                                                                                            eval_results['ep_lengths'].std(),
                                                                                                            eval_results['ep_returns'].mean(),
                                                                                                            eval_results['ep_returns'].std()))
                # Save best model.
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))
            # Logging.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)
    
    def _learn(self,
               env=None,
               **kwargs
               ):
        '''Performs learning as an unified calling function for hyperparameter optimization.

        Args:
            env (BenchmarkEnv): The environment to be used for training.
        '''
        return self.learn(env=env, **kwargs)

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
            action = self.agent.ac.act(obs)

        return action

    def run(self,
            env=None,
            render=False,
            n_episodes=10,
            verbose=False,
            ):
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
                obs, _ = env.reset()
            obs = self.obs_normalizer(obs)
        # Collect evaluation results.
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
    
    def _run(self, **kwargs):
        '''Runs evaluation as an unified calling function for hyperparameter optimization.
        '''
        results = self.run(env=self.eval_env, render=False, n_episodes=self.eval_batch_size, verbose=False, **kwargs)
        mean_cost = np.mean(results["ep_returns"])

        return mean_cost
    
    def _eval_combined_policy(self, h=0):
        """Evaluate the combined policy containing guide policy and explore policy,
           i.e., the rewards of pi = pi_g[:h] + pi_e[h:H], where H is the control horizon.

           args: h (int): guide step

           returns: combined_policy_reward (float): the reward of combined policy
                    explore_policy_reward (float): the reward of explore policy
                    expected_explore_policy_reward (float): how much reward of explore policy is expected 
                    guide_control_horizon (int): the control horizon of pure guide policy

        """
        self.prior_ctrl.x_prev = None
        self.prior_ctrl.u_prev = None
        if not self._prior_eval_env.initial_reset:
            self._prior_eval_env.set_cost_function_param(self.prior_ctrl.Q, self.prior_ctrl.R)

        obs, info = self._prior_eval_env.reset()
        obs = self.obs_normalizer(obs)

        # state dimension
        nx = self._prior_eval_env.state.shape[0]

        if self._prior_eval_env.TASK == Task.STABILIZATION:
            MAX_STEPS = int(self._prior_eval_env.CTRL_FREQ * self._prior_eval_env.EPISODE_LEN_SEC)
        elif self._prior_eval_env.TASK == Task.TRAJ_TRACKING:
            MAX_STEPS = self.prior_ctrl.traj.shape[1]

        self.prior_ctrl.terminate_loop = False
        done = False
        done_sync = True
        terminate_run_on_done = True
        i = 0
        guide_policy_reward_until_h = 0
        guide_policy_reward = 0
        while not(done and terminate_run_on_done) and i < MAX_STEPS and not (self.prior_ctrl.terminate_loop):
            pure_obs = obs[:nx]
            action = self.prior_ctrl.select_action(pure_obs)
            if self.prior_ctrl.terminate_loop:
                print("Infeasible MPC Problem")
                break

            # normalize action for rl env
            action_rl = action/self._prior_eval_env.action_scale
            obs, reward, done, info = self._prior_eval_env.step(action_rl)
            obs = self.obs_normalizer(obs)
            pure_obs = obs[:nx]

            guide_policy_reward += reward
            if i == h:
                # compute the guide policy reward untill h-step
                guide_policy_reward_until_h = guide_policy_reward
                # save the env state at h-step
                eval_env = deepcopy(self._prior_eval_env)
                obs_sync = obs
                info_sync = info
                done_sync = done

            i += 1
        guide_control_horizon = i

        # explore policy take over the decision making after h-th step
        explore_policy_reward = 0
        i = 0
        while done_sync == False:
            action = self.select_action(obs=obs_sync, info=info_sync)
            obs_sync, reward, done_sync, info_sync = eval_env.step(action)
            obs_sync = self.obs_normalizer(obs_sync)
            explore_policy_reward += reward

            i += 1

        obs = self._prior_eval_env.reset()

        combined_policy_reward = guide_policy_reward_until_h + explore_policy_reward
        expected_explore_policy_reward = guide_policy_reward - guide_policy_reward_until_h

        return combined_policy_reward, explore_policy_reward, expected_explore_policy_reward, guide_control_horizon

    def train_step(self):
        '''Performs a training/fine-tuning step.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        rollouts = PPOBuffer(self.env.observation_space, self.env.action_space, self.rollout_steps, self.rollout_batch_size)
        obs = self.obs
        start = time.time()
        for _ in range(self.rollout_steps):
            with torch.no_grad():
                act, v, logp = self.agent.ac.step(torch.FloatTensor(obs).to(self.device))
            next_obs, rew, done, info = self.env.step(act)
            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            mask = 1 - done.astype(float)
            # Time truncation is not the same as true termination.
            terminal_v = np.zeros_like(v)
            for idx, inf in enumerate(info['n']):
                if 'terminal_info' not in inf:
                    continue
                inff = inf['terminal_info']
                if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                    terminal_obs = inf['terminal_observation']
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).unsqueeze(0).to(self.device)
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().cpu().numpy()
                    terminal_v[idx] = terminal_val
            rollouts.push({'obs': obs, 'act': act, 'rew': rew, 'mask': mask, 'v': v, 'logp': logp, 'terminal_v': terminal_v})
            obs = next_obs
        self.obs = obs
        self.total_steps += self.rollout_batch_size * self.rollout_steps
        # Learn from rollout batch.
        last_val = self.agent.ac.critic(torch.FloatTensor(obs).to(self.device)).detach().cpu().numpy()
        ret, adv = compute_returns_and_advantages(rollouts.rew,
                                                  rollouts.v,
                                                  rollouts.mask,
                                                  rollouts.terminal_v,
                                                  last_val,
                                                  gamma=self.gamma,
                                                  use_gae=self.use_gae,
                                                  gae_lambda=self.gae_lambda)
        rollouts.ret = ret
        # Prevent divide-by-0 for repetitive tasks.
        rollouts.adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        results = self.agent.update(rollouts, self.device)
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def log_step(self,
                 results
                 ):
        '''Does logging after a training step.'''
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'step_time': results['elapsed_time'],
                'progress': step / self.max_env_steps
            },
            step,
            prefix='time')
        # Learning stats.
        self.logger.add_scalars(
            {
                k: results[k]
                for k in ['policy_loss', 'value_loss', 'entropy_loss', 'approx_kl']
            },
            step,
            prefix='loss')
        # Performance stats.
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
        # Total constraint violation during learning.
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
        # Print summary table
        self.logger.dump_scalars()
