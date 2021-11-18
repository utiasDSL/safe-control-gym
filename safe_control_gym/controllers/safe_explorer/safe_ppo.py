"""PPO-based safe explorer.

"""
import os
import time
import numpy as np
import torch
from collections import defaultdict

from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, set_random_state, is_wrapped
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_obs
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics
from safe_control_gym.math_and_models.normalization import BaseNormalizer, MeanStdNormalizer, RewardStdNormalizer

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.ppo.ppo_utils import compute_returns_and_advantages
from safe_control_gym.controllers.safe_explorer.safe_explorer_utils import SafetyLayer, ConstraintBuffer
from safe_control_gym.controllers.safe_explorer.safe_ppo_utils import SafePPOAgent, SafePPOBuffer


class SafeExplorerPPO(BaseController):
    """Safety layer for constraint satisfaction for RL.

    """

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path="model_latest.pt",
                 output_dir="temp",
                 device="cpu",
                 seed=0,
                 **kwargs
                 ):
        super().__init__(env_func, training, checkpoint_path, output_dir, device, seed, **kwargs)
        # Task.
        if self.training:
            # Training and testing.
            self.env = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.env = VecRecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
            self.num_constraints = self.env.envs[0].num_constraints
        else:
            # Testing only.
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)
            self.num_constraints = self.env.num_constraints
        # Safety layer.
        self.safety_layer = SafetyLayer(self.env.observation_space,
                                        self.env.action_space,
                                        hidden_dim=self.constraint_hidden_dim,
                                        num_constraints=self.num_constraints,
                                        lr=self.constraint_lr,
                                        slack=self.constraint_slack)
        self.safety_layer.to(device)
        # Agent.
        self.agent = SafePPOAgent(
            self.env.observation_space,
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
            action_modifier=self.safety_layer.get_safe_action,
        )
        self.agent.to(device)
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
            # Disable logging to texts and tfboard for evaluation.
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def reset(self):
        """Do initializations for training or evaluation.

        """
        if self.training:
            if self.pretraining:
                self.constraint_buffer = ConstraintBuffer(self.env.observation_space, self.env.action_space, self.num_constraints, self.constraint_buffer_size)
            else:
                # Load safety layer for 2nd stage training.
                assert self.pretrained, "Must provide a pre-trained model for adaptation."
                if os.path.isdir(self.pretrained):
                    self.pretrained = os.path.join(self.pretrained, "model_latest.pt")
                state = torch.load(self.pretrained)
                self.safety_layer.load_state_dict(state["safety_layer"])
                # Set up stats tracking.
                self.env.add_tracker("constraint_violation", 0)
                self.env.add_tracker("constraint_violation", 0, mode="queue")
                self.eval_env.add_tracker("constraint_violation", 0, mode="queue")
                self.eval_env.add_tracker("mse", 0, mode="queue")
            self.total_steps = 0
            obs, info = self.env.reset()
            self.obs = self.obs_normalizer(obs)
            self.c = np.array([inf["constraint_values"] for inf in info["n"]])
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
            "safety_layer": self.safety_layer.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
        }
        if self.training:
            exp_state = {
                "total_steps": self.total_steps,
                "obs": self.obs,
                "c": self.c,
                "random_state": get_random_state(),
                "env_random_state": self.env.get_env_random_state()
            }
            state_dict.update(exp_state)
            if self.pretraining:
                state_dict["constraint_buffer"] = self.constraint_buffer.state_dict()
        torch.save(state_dict, path)

    def load(self,
             path
             ):
        """Restores model and experiment given checkpoint path.

        """
        state = torch.load(path)
        # Restore policy.
        self.agent.load_state_dict(state["agent"])
        self.safety_layer.load_state_dict(state["safety_layer"])
        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])
        # Restore experiment state.
        if self.training:
            self.total_steps = state["total_steps"]
            self.obs = state["obs"]
            self.c = state["c"]
            set_random_state(state["random_state"])
            self.env.set_env_random_state(state["env_random_state"])
            self.logger.load(self.total_steps)
            if self.pretraining:
                self.constraint_buffer.load_state_dict(state["constraint_buffer"])

    def learn(self,
              env=None,
              **kwargs
              ):
        """Performs learning (pre-training, training, fine-tuning, etc).

        """
        if self.pretraining:
            final_step = self.constraint_epochs
            train_func = self.pretrain_step
        else:
            final_step = self.max_env_steps
            train_func = self.train_step
        while self.total_steps < final_step:
            results = train_func()
            # Checkpoint.
            if self.total_steps >= final_step or (self.save_interval and self.total_steps % self.save_interval == 0):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info("Checkpoint | {}".format(self.checkpoint_path))
            if self.num_checkpoints and self.total_steps % (final_step // self.num_checkpoints) == 0:
                # Intermediate checkpoint.
                path = os.path.join(self.output_dir, "checkpoints", "model_{}.pt".format(self.total_steps))
                self.save(path)
            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                if self.pretraining:
                    eval_results = self.eval_constraint_models()
                    results["eval"] = eval_results
                else:
                    eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                    results["eval"] = eval_results
                    self.logger.info("Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}".format(eval_results["ep_lengths"].mean(),
                                                                                                                eval_results["ep_lengths"].std(),
                                                                                                                eval_results["ep_returns"].mean(),
                                                                                                                eval_results["ep_returns"].std()))
                    # Save the best model.
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
                env.add_tracker("constraint_violation", 0, mode="queue")
                env.add_tracker("constraint_values", 0, mode="queue")
                env.add_tracker("mse", 0, mode="queue")
        obs, info = env.reset()
        obs = self.obs_normalizer(obs)
        c = info["constraint_values"]
        ep_returns, ep_lengths = [], []
        frames = []
        while len(ep_returns) < n_episodes:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(self.device)
                c = torch.FloatTensor(c).to(self.device)
                action = self.agent.ac.act(obs, c=c)
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
                obs, info = env.reset()
            obs = self.obs_normalizer(obs)
            c = info["constraint_values"]
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

    def pretrain_step(self):
        """Performs a pre-trianing step.

        """
        results = defaultdict(list)
        start = time.time()
        self.safety_layer.train()
        self.obs_normalizer.unset_read_only()
        # Just sample episodes for the whole epoch.
        self.collect_constraint_data(self.constraint_steps_per_epoch)
        self.total_steps += 1
        # Do the update from memory.
        for batch in self.constraint_buffer.sampler(self.constraint_batch_size):
            res = self.safety_layer.update(batch)
            for k, v in res.items():
                results[k].append(v)
        self.constraint_buffer.reset()
        results = {k: sum(v) / len(v) for k, v in results.items()}
        results.update({"step": self.total_steps, "elapsed_time": time.time() - start})
        return results

    def train_step(self):
        """Performs a training/fine-tuning step.

        """
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        rollouts = SafePPOBuffer(self.env.observation_space,
                                 self.env.action_space,
                                 self.num_constraints,
                                 self.rollout_steps,
                                 self.rollout_batch_size)
        obs = self.obs
        c = self.c
        start = time.time()
        for step in range(self.rollout_steps):
            with torch.no_grad():
                act, v, logp = self.agent.ac.step(torch.FloatTensor(obs).to(self.device), c=torch.FloatTensor(c).to(self.device))
            next_obs, rew, done, info = self.env.step(act)
            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            mask = 1 - done.astype(float)
            # Time truncation is not the same as the true termination.
            terminal_v = np.zeros_like(v)
            for idx, inf in enumerate(info["n"]):
                if "terminal_info" not in inf:
                    continue
                inff = inf["terminal_info"]
                if "TimeLimit.truncated" in inff and inff["TimeLimit.truncated"]:
                    terminal_obs = inf["terminal_observation"]
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).unsqueeze(0).to(self.device)
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().numpy()
                    terminal_v[idx] = terminal_val
            rollouts.push({
                "obs": obs,
                "act": act,
                "rew": rew,
                "mask": mask,
                "v": v,
                "logp": logp,
                "terminal_v": terminal_v,
                "c": c,
            })
            obs = next_obs
            c = np.array([inf["constraint_values"] for inf in info["n"]])
        self.obs = obs
        self.c = c
        self.total_steps += self.rollout_batch_size * self.rollout_steps
        # Learn from rollout batch.
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
        final_step = self.constraint_epochs if self.pretraining else self.max_env_steps
        # Runner stats.
        self.logger.add_scalars(
            {
                "step": step,
                "time": results["elapsed_time"],
                "progress": step / final_step
            },
            step,
            prefix="time",
            write=False,
            write_tb=False)
        if self.pretraining:
            # Constraint learning stats.
            for i in range(self.safety_layer.num_constraints):
                name = "constraint_{}_loss".format(i)
                self.logger.add_scalars({name: results[name]}, step, prefix="constraint_loss")
                if "eval" in results:
                    self.logger.add_scalars({name: results["eval"][name]}, step, prefix="constraint_loss_eval")
        else:
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
        # Print summary table.
        self.logger.dump_scalars()

    def collect_constraint_data(self,
                                num_steps
                                ):
        """Uses random policy to collect data for pre-training constriant models.

        """
        step = 0
        obs, info = self.env.reset()
        obs = self.obs_normalizer(obs)
        c = np.array([inf["constraint_values"] for inf in info["n"]])
        while step < num_steps:
            action_spaces = self.env.get_attr("action_space")
            action = np.array([space.sample() for space in action_spaces])
            obs_next, _, done, info = self.env.step(action)
            obs_next = self.obs_normalizer(obs_next)
            c_next = []
            for i, d in enumerate(done):
                if d:
                    c_next_i = info["n"][i]["terminal_info"]["constraint_values"]
                else:
                    c_next_i = info["n"][i]["constraint_values"]
                c_next.append(c_next_i)
            c_next = np.array(c_next)
            self.constraint_buffer.push({"act": action, "obs": obs, "c": c, "c_next": c_next})
            obs = obs_next
            c = np.array([inf["constraint_values"] for inf in info["n"]])
            step += self.rollout_batch_size

    def eval_constraint_models(self):
        """Runs evaluation for the constraint models.

        """
        eval_resutls = defaultdict(list)
        self.safety_layer.eval()
        self.obs_normalizer.set_read_only()
        # Collect evaluation data.
        self.collect_constraint_data(self.constraint_eval_steps)
        for batch in self.constraint_buffer.sampler(self.constraint_batch_size):
            losses = self.safety_layer.compute_loss(batch)
            for i, loss in enumerate(losses):
                eval_resutls["constraint_{}_loss".format(i)].append(loss.item())
        self.constraint_buffer.reset()
        eval_resutls = {k: sum(v) / len(v) for k, v in eval_resutls.items()}
        return eval_resutls
