"""Proximal Policy Optimization (PPO)

Based on:
    * https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    * (hyperparameters) https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

Additional references:
    * Proximal Policy Optimization Algorithms - https://arxiv.org/pdf/1707.06347.pdf
    * Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO - https://arxiv.org/pdf/2005.12729.pdf
    * pytorch-a2c-ppo-acktr-gail - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    * openai spinning up - ppo - https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
    * stable baselines3 - ppo - https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo
    
"""
import os
import time
import numpy as np
import torch
from collections import defaultdict

from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, set_random_state, is_wrapped
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics
from safe_control_gym.math_and_models.normalization import BaseNormalizer, MeanStdNormalizer, RewardStdNormalizer

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.ppo.ppo_utils import PPOAgent, PPOBuffer, compute_returns_and_advantages


class PPO(BaseController):
    """Proximal policy optimization.

    """

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path="model_latest.pt",
                 output_dir="temp",
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        # Task.
        if self.training:
            # Training and testing.
            self.env = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.env = VecRecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
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
                              mini_batch_size=self.mini_batch_size)
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
        """Do initializations for training or evaluation.

        """
        if self.training:
            # set up stats tracking
            self.env.add_tracker("constraint_violation", 0)
            self.env.add_tracker("constraint_violation", 0, mode="queue")
            self.eval_env.add_tracker("constraint_violation", 0, mode="queue")
            self.eval_env.add_tracker("mse", 0, mode="queue")
            
            self.total_steps = 0
            obs, _ = self.env.reset()
            self.obs = self.obs_normalizer(obs)
        else:
            # Add episodic stats to be tracked.
            self.env.add_tracker("constraint_violation", 0, mode="queue")
            self.env.add_tracker("constraint_values", 0, mode="queue")
            self.env.add_tracker("mse", 0, mode="queue")

    def close(self):
        """Shuts down and cleans up lingering resources.

        """
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self,
             path
             ):
        """Saves model params and experiment state to checkpoint path.

        """
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        state_dict = {
            "agent": self.agent.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
        }
        if self.training:
            exp_state = {
                "total_steps": self.total_steps,
                "obs": self.obs,
                "random_state": get_random_state(),
                "env_random_state": self.env.get_env_random_state()
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self,
             path
             ):
        """Restores model and experiment given checkpoint path.

        """
        state = torch.load(path)
        # Restore policy.
        self.agent.load_state_dict(state["agent"])
        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])
        # Restore experiment state.
        if self.training:
            self.total_steps = state["total_steps"]
            self.obs = state["obs"]
            set_random_state(state["random_state"])
            self.env.set_env_random_state(state["env_random_state"])
            self.logger.load(self.total_steps)

    def learn(self,
              env=None,
              **kwargs
              ):
        """Performs learning (pre-training, training, fine-tuning, etc).

        """
        while self.total_steps < self.max_env_steps:
            results = self.train_step()
            # Checkpoint.
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info("Checkpoint | {}".format(self.checkpoint_path))
            if self.num_checkpoints and self.total_steps % (self.max_env_steps // self.num_checkpoints) == 0:
                # Intermediate checkpoint.
                path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
                self.save(path)
            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results["eval"] = eval_results
                self.logger.info("Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}".format(eval_results["ep_lengths"].mean(),
                                                                                                            eval_results["ep_lengths"].std(),
                                                                                                            eval_results["ep_returns"].mean(),
                                                                                                            eval_results["ep_returns"].std()))
                # Save best model.
                eval_score = eval_results["ep_returns"].mean()
                eval_best_score = getattr(self, "eval_best_score", -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, "model_best.pt"))
            # Logging.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def run(self,
            env=None,
            render=False,
            n_episodes=10,
            verbose=False,
            **kwargs
            ):
        """Runs evaluation with current policy.

        """
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add eposodic stats to be tracked.
                env.add_tracker("constraint_violation", 0, mode="queue")
                env.add_tracker("constraint_values", 0, mode="queue")
                env.add_tracker("mse", 0, mode="queue")
                
        obs, info = env.reset()
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        while len(ep_returns) < n_episodes:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device)
                action = self.agent.ac.act(obs)
            obs, reward, done, info = env.step(action)
            if render:
                env.render()
                frames.append(env.render("rgb_array"))
            if verbose:
                print("obs {} | act {}".format(obs, action))
            if done:
                assert "episode" in info
                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])
                obs, _ = env.reset()
            obs = self.obs_normalizer(obs)
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {"ep_returns": ep_returns, "ep_lengths": ep_lengths}
        if len(frames) > 0:
            eval_results["frames"] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def train_step(self):
        """Performs a training/fine-tuning step.

        """
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        rollouts = PPOBuffer(self.env.observation_space, self.env.action_space, self.rollout_steps, self.rollout_batch_size)
        obs = self.obs
        start = time.time()
        for step in range(self.rollout_steps):
            with torch.no_grad():
                act, v, logp = self.agent.ac.step(torch.FloatTensor(obs).to(self.device))
            next_obs, rew, done, info = self.env.step(act)
            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            mask = 1 - done.astype(float)
            # Time truncation is not the same as true termination.
            terminal_v = np.zeros_like(v)
            for idx, inf in enumerate(info["n"]):
                if "terminal_info" not in inf:
                    continue
                inff = inf["terminal_info"]
                if "TimeLimit.truncated" in inff and inff["TimeLimit.truncated"]:
                    terminal_obs = inf["terminal_observation"]
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).unsqueeze(0).to(self.device)
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().cpu().numpy()
                    terminal_v[idx] = terminal_val
            rollouts.push({"obs": obs, "act": act, "rew": rew, "mask": mask, "v": v, "logp": logp, "terminal_v": terminal_v})
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
        results.update({"step": self.total_steps, "elapsed_time": time.time() - start})
        return results

    def log_step(self,
                 results
                 ):
        """Does logging after a training step.

        """
        step = results["step"]
        # runner stats
        self.logger.add_scalars(
            {
                "step": step,
                "step_time": results["elapsed_time"],
                "progress": step / self.max_env_steps
            },
            step,
            prefix="time",
            write=False,
            write_tb=False)
        # Learning stats.
        self.logger.add_scalars(
            {
                k: results[k] 
                for k in ["policy_loss", "value_loss", "entropy_loss", "approx_kl"]
            }, 
            step, 
            prefix="loss")
        # Performance stats.
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        ep_constraint_violation = np.asarray(self.env.queued_stats["constraint_violation"])
        self.logger.add_scalars(
            {
                "ep_length": ep_lengths.mean(),
                "ep_return": ep_returns.mean(),
                "ep_reward": (ep_returns / ep_lengths).mean(),
                "ep_constraint_violation": ep_constraint_violation.mean()
            },
            step,
            prefix="stat")
        # Total constraint violation during learning.
        total_violations = self.env.accumulated_stats["constraint_violation"]
        self.logger.add_scalars({"constraint_violation": total_violations}, step, prefix="stat")
        if "eval" in results:
            eval_ep_lengths = results["eval"]["ep_lengths"]
            eval_ep_returns = results["eval"]["ep_returns"]
            eval_constraint_violation = results["eval"]["constraint_violation"]
            eval_mse = results["eval"]["mse"]
            self.logger.add_scalars(
                {
                    "ep_length": eval_ep_lengths.mean(),
                    "ep_return": eval_ep_returns.mean(),
                    "ep_reward": (eval_ep_returns / eval_ep_lengths).mean(),
                    "constraint_violation": eval_constraint_violation.mean(),
                    "mse": eval_mse.mean()
                },
                step,
                prefix="stat_eval")
        # Print summary table
        self.logger.dump_scalars()
