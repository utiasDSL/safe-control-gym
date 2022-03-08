"""Robust Adversarial Reinforcement Learning using Adversarial Populations (RAP)

References papers & code:
    * [Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702)
    * [Robust Reinforcement Learning using Adversarial Populations](https://arxiv.org/abs/2008.01825)
    * [robust-adversarial-rl](https://github.com/jerinphilip/robust-adversarial-rl)
    * [rllab-adv](https://github.com/lerrel/rllab-adv)
    * [Robust Reinforcement Learning via adversary pools](https://github.com/eugenevinitsky/robust_RL_multi_adversary)

Example: 
    train on cartpole_adversary::
    
        $ python tests/test_main.py --mode train_two_phase --exp_id rap_cartpole_adv \
        --algo rap --task cartpole_adversary --num_workers 2 --max_env_steps 2000000 \
        --tensorboard --use_gae --num_adversaries 2

Todo:
    *

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
from safe_control_gym.controllers.rarl.rarl_utils import split_obs_by_adversary


class RAP(BaseController):
    """rarl via adersarial population with PPO."""

    def __init__(self, 
                 env_func, 
                 training=True, 
                 checkpoint_path="model_latest.pt", 
                 output_dir="temp", 
                 use_gpu=False, 
                 seed=0, 
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        self.use_gpu = use_gpu
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

        # protagonist and adversary agents
        shared_agent_args = dict(hidden_dim=self.hidden_dim,
                                 use_clipped_value=self.use_clipped_value,
                                 clip_param=self.clip_param,
                                 target_kl=self.target_kl,
                                 entropy_coef=self.entropy_coef,
                                 actor_lr=self.actor_lr,
                                 critic_lr=self.critic_lr,
                                 opt_epochs=self.opt_epochs,
                                 mini_batch_size=self.mini_batch_size)

        self.agent = PPOAgent(self.env.observation_space, self.env.action_space, **shared_agent_args)
        self.agent.to(self.device)

        # fetch adversary specs from env 
        if self.training:
            self.adv_obs_space = self.env.get_attr("adversary_observation_space")[0]
            self.adv_act_space = self.env.get_attr("adversary_action_space")[0]
        else:
            self.adv_obs_space = self.env.adversary_observation_space
            self.adv_act_space = self.env.adversary_action_space
        self.adversaries = [PPOAgent(self.adv_obs_space, self.adv_act_space, **shared_agent_args) for _ in range(self.num_adversaries)]
        for adv in self.adversaries:
            adv.to(self.device)

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
            # disable logging to texts and tfboard for evaluation
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        """Do initializations for training or evaluation."""
        if self.training:
            # Add episodic stats to be tracked.
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
        """Shuts down and cleans up lingering resources."""
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self, path):
        """Saves model params and experiment state to checkpoint path."""
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)

        state_dict = {
            "agent": self.agent.state_dict(),
            "adversary": [adv.state_dict() for adv in self.adversaries],
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

    def load(self, path):
        """Restores model and experiment given checkpoint path."""
        state = torch.load(path)

        # restore pllicy
        self.agent.load_state_dict(state["agent"])
        for i, adv_state_dict in enumerate(state["adversary"]):
            self.adversaries[i].load_state_dict(adv_state_dict)
        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])

        # restore experiment state
        if self.training:
            self.total_steps = state["total_steps"]
            self.obs = state["obs"]
            set_random_state(state["random_state"])
            self.env.set_env_random_state(state["env_random_state"])
            self.logger.load(self.total_steps)

    def learn(self, env=None, **kwargs):
        """Performs learning (pre-training, training, fine-tuning, etc)."""
        while self.total_steps < self.max_env_steps:
            results = self.train_step()

            # checkpoint
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # latest/final checkpoint
                self.save(self.checkpoint_path)
                self.logger.info("Checkpoint | {}".format(self.checkpoint_path))
            if self.num_checkpoints and self.total_steps % (self.max_env_steps // self.num_checkpoints) == 0:
                # intermediate checkpoint
                path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
                self.save(path)

            # eval
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results["eval"] = eval_results
                self.logger.info("Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}".format(eval_results["ep_lengths"].mean(),
                                                                                                            eval_results["ep_lengths"].std(),
                                                                                                            eval_results["ep_returns"].mean(),
                                                                                                            eval_results["ep_returns"].std()))
                # save best model
                eval_score = eval_results["ep_returns"].mean()
                eval_best_score = getattr(self, "eval_best_score", -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, "model_best.pt"))

            # logging
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def run(self, env=None, render=False, n_episodes=10, verbose=False, use_adv=False, **kwargs):
        """Runs evaluation with current policy."""
        self.agent.eval()
        for adv in self.adversaries:
            adv.eval()
        self.obs_normalizer.set_read_only()
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                env.add_tracker("constraint_violation", 0, mode="queue")
                env.add_tracker("constraint_values", 0, mode="queue")
                env.add_tracker("mse", 0, mode="queue")

        obs, _ = env.reset()
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []

        while len(ep_returns) < n_episodes:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device)
                action = self.agent.ac.act(obs)

            # no disturbance during testing
            if use_adv:
                adv_idx = np.random.choice(self.num_adversaries)
                with torch.no_grad():
                    action_adv = self.adversaries[adv_idx].ac.act(obs)
            else:
                action_adv = np.zeros(self.adv_act_space.shape[0])
            env.set_adversary_control(action_adv)

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

        # collect evaluation results
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
        """Performs a training/fine-tuning step."""
        self.agent.train()
        for adv in self.adversaries:
            adv.train()
        self.obs_normalizer.unset_read_only()
        start = time.time()
        results = defaultdict(list)

        # collect trajectories (with different adversary each time)
        rollouts, rollout_splits = self.collect_rollouts()

        # perform updates for both agent and adversaries
        agent_results = self.agent.update(rollouts)
        results.update(agent_results)

        for adv_idx, adv_rollouts in rollout_splits:
            adv_results = self.adversaries[adv_idx].update(adv_rollouts)
            adv_results = {k + "_adv{}".format(adv_idx): v for k, v in adv_results.items()}
            results.update(adv_results)

        # miscellaneous
        results.update({"step": self.total_steps, "elapsed_time": time.time() - start, "adv_indices": [adv_idx for adv_idx, _ in rollout_splits]})
        return results

    def log_step(self, results):
        """Does logging after a training step."""
        step = results["step"]
        # runner stats
        self.logger.add_scalars(
            {
                "step": step,
                "time": results["elapsed_time"],
                "progress": step / self.max_env_steps
            },
            step,
            prefix="time",
            write=False,
            write_tb=False)

        # learning stats
        self.logger.add_scalars(
            {
                k: results[k] 
                for k in ["policy_loss", "value_loss", "entropy_loss"]
            }, 
            step, 
            prefix="loss")
        for adv_idx in results["adv_indices"]:
            self.logger.add_scalars(
                {
                    k: results[k + "_adv{}".format(adv_idx)] 
                    for k in ["policy_loss", "value_loss", "entropy_loss"]
                },
                step,
                prefix="loss_adv{}".format(adv_idx))

        # performance stats
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
        # print summary table
        self.logger.dump_scalars()

    def collect_rollouts(self):
        """Gets trajectories (full episodes) for both agent and adversaries."""
        # agent & adversary must have same obs & act space
        rollouts = PPOBuffer(self.env.observation_space, self.env.action_space, self.rollout_steps, self.rollout_batch_size)
        rollouts_adv = PPOBuffer(self.adv_obs_space, self.adv_act_space, self.rollout_steps, self.rollout_batch_size)

        # sample adversaries
        adv_indices = np.random.randint(self.num_adversaries, size=self.rollout_batch_size)
        adv_indices.sort()
        indices_groups, indices_splits = np.unique(adv_indices, return_index=True)

        # sample trajectories
        # TODO: fix it, never finish a full trajectory
        obs, _ = self.env.reset()
        obs = self.obs_normalizer(obs)

        for step in range(self.rollout_steps):
            # get actions
            with torch.no_grad():
                act, v, logp = self.agent.ac.step(torch.FloatTensor(obs).to(self.device))

                # adversary actions
                obs_groups = split_obs_by_adversary(obs, indices_splits)
                out_adv = []
                for idx, obs_adv in zip(indices_groups, obs_groups):
                    out = self.adversaries[idx].ac.step(torch.FloatTensor(obs_adv).to(self.device))
                    out_adv.append(out)
                act_adv, v_adv, logp_adv = [np.concatenate(item) for item in zip(*out_adv)]

            # step env
            # self.env.set_adversary_control(act_adv)
            act_adv_list = [[act] for act in act_adv]
            self.env.env_method("set_adversary_control", act_adv_list)
            next_obs, rew, done, info = self.env.step(act)

            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            mask = 1 - done.astype(float)

            # time truncation is not true termination
            terminal_v = np.zeros_like(v)
            terminal_v_adv = np.zeros_like(v_adv)
            for idx, inf in enumerate(info["n"]):
                # if "TimeLimit.truncated" in inf and inf["TimeLimit.truncated"]:
                if "terminal_info" not in inf:
                    continue
                inff = inf["terminal_info"]
                if "TimeLimit.truncated" in inff and inff["TimeLimit.truncated"]:
                    terminal_obs = inf["terminal_observation"]
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).unsqueeze(0).to(self.device)

                    # estimate value for terminated state
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().numpy()
                    terminal_v[idx] = terminal_val

                    # estimate terminal value for adversary
                    adversary = self.adversaries[adv_indices[idx]]
                    terminal_val_adv = adversary.ac.critic(terminal_obs_tensor).squeeze().detach().numpy()
                    terminal_v_adv[idx] = terminal_val_adv

            # collect rollout data
            rollouts.push({"obs": obs, "act": act, "rew": rew, "mask": mask, "v": v, "logp": logp, "terminal_v": terminal_v})
            # no need to push `obs`, `mask` since they are the same
            rollouts_adv.push({
                "act": act_adv,
                "rew": -rew,
                "v": v_adv,
                "logp": logp_adv,
                "terminal_v": terminal_v_adv,
            })
            obs = next_obs

        # bookkeep
        self.total_steps += self.rollout_batch_size * self.rollout_steps

        # postprocess for main agent
        last_val = self.agent.ac.critic(torch.FloatTensor(obs).to(self.device)).detach().numpy()
        ret, adv = compute_returns_and_advantages(rollouts.rew,
                                                  rollouts.v,
                                                  rollouts.mask,
                                                  rollouts.terminal_v,
                                                  last_val,
                                                  gamma=self.gamma,
                                                  use_gae=self.use_gae,
                                                  gae_lambda=self.gae_lambda)
        rollouts.ret = ret
        rollouts.adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # postprocess for adversary
        rollouts_adv.obs = rollouts.obs
        rollouts_adv.mask = rollouts.mask

        obs_groups = split_obs_by_adversary(obs, indices_splits)
        last_val_adv = []
        for idx, obs_adv in zip(indices_groups, obs_groups):
            out = self.adversaries[idx].ac.critic(torch.FloatTensor(obs_adv).to(self.device)).detach().numpy()
            last_val_adv.append(out)
        last_val_adv = np.concatenate(last_val_adv)

        ret, adv = compute_returns_and_advantages(rollouts_adv.rew,
                                                  rollouts_adv.v,
                                                  rollouts_adv.mask,
                                                  rollouts_adv.terminal_v,
                                                  last_val_adv,
                                                  gamma=self.gamma,
                                                  use_gae=self.use_gae,
                                                  gae_lambda=self.gae_lambda)
        rollouts_adv.ret = ret
        rollouts_adv.adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # split adversary rollouts for each
        rollout_splits = []
        start = indices_splits
        end = np.concatenate([indices_splits[1:], [self.rollout_batch_size]])

        for idx, s_idx, e_idx in zip(indices_groups, start, end):
            split_batch_size = e_idx - s_idx
            rollout_split = PPOBuffer(self.adv_obs_space, self.adv_act_space, self.rollout_steps, split_batch_size)
            for k in rollouts_adv.scheme:
                # rollout_split[k] = rollouts_adv[k][:, s_idx:e_idx]
                setattr(rollout_split, k, getattr(rollouts_adv, k)[:, s_idx:e_idx])
            rollout_splits.append([idx, rollout_split])

        return rollouts, rollout_splits