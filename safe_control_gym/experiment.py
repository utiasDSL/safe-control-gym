"""To standardize training/evaluation interface.

Todo:
    * 
    
"""
import os 
import gym
import numpy as np 
from copy import deepcopy
import functools
from collections import defaultdict, deque
from termcolor import colored

# from safe_control_gym.utils.configuration import ConfigFactory
# from safe_control_gym.utils.registration import make
# from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video
from safe_control_gym.utils.utils import get_random_state, set_random_state, is_wrapped
from safe_control_gym.math_and_models.metrics import compute_cvar



class Experiment:
    
    def __init__(self):
        # what to put here? 
        # now the class seems like a collection of utils (from `main.py`) only.
        pass
    
    def launch_training(self, config, ctrl):
        """Since the learning loop varies among controllers, can only delegate to its own `learn()` method.

        Note ctrl should have its own training env as attribute, which is constructed upon creating the ctrl. 
        We do not standarize on the training env as input to this method since different controllers use it differently. 
        """
        ctrl.reset()
        if config.restore:
            ctrl.load(os.path.join(config.restore, "model_latest.pt"))
        ctrl.learn()
        print("Training done.")
    
    def run_evaluation(self, config, ctrl, env, n_episodes=10, **kwargs):
        """Evaluate a trained controller.

        Note this evaluation uses the ctrl's own control frequency.
        Optionally, use the ctrl's own `run()` method to evaluate instead of collecting full traj data.
        """
        ctrl.reset()
        if config.restore:
            ctrl.load(os.path.join(config.restore, "model_latest.pt"))
        trajs_data = self.collect_trajs(ctrl, env, n_trajs=n_episodes)
        results = self.compute_metrics(trajs_data)
        # terminal printouts
        for metric_key, metric_val in results.items():
            print("{}: {:.3f}".format(colored(metric_key,"yellow"), metric_val))
        print("Evaluation done.")
        return results
    
    def collect_trajs(self, ctrl, env, log_freq=None, n_trajs=None, n_steps=None, **kwargs):
        """"""
        if not is_wrapped(env, RecordDataWrapper):
            env = RecordDataWrapper(env)
        
        # initialize
        sim_steps = log_freq // env.CTRL_FREQ if log_freq else 1 
        n_step, n_traj = 0, 0
        done = False
        ctrl_data = defaultdict(list)
        if env.INFO_IN_RESET:
            obs, info = env.reset()
        else:
            obs = env.reset()
            info = None
        # cache anything the ctrl needs for running an episode later
        ctrl.reset_before_run(obs, info, env=env)
        for data_key, data_val in ctrl.get_eval_result_dict():
            ctrl_data[data_key].append(data_val)
            
        # collect data 
        while True:
            # everything other than `obs` that the ctrl needs should be in info, 
            # or otherwise should be kept as ctrl attributes 
            act = ctrl.select_action(obs, info)
            # inner sim loop to accomodate different control frequencies
            for _ in range(sim_steps):
                obs, rew, done, info = env.step(act)
                n_step += 1 
                if done:
                    n_traj += 1
                    done = False
                    if env.INFO_IN_RESET:
                        obs, info = env.reset()
                    else:
                        obs = env.reset()
                        info = None
                    ctrl.reset_before_run(obs, info, env=env)
                    break
                # terminate when data is enough
            if (n_trajs and n_traj >= n_trajs) or (n_steps and n_step >= n_steps):
                break
        
        # compile collected data 
        results = env.data 
        results.update(ctrl.get_eval_result_dict())
        return results
    
    def compute_metrics(self, trajs_data, **kwargs):
        """"""
        met = MetricExtractor(trajs_data)
        # collect & compute all sorts of metrics here
        results = {
            "average_length": np.asarray(met.get_episode_lengths).mean(),
            "average_return": np.asarray(met.get_episode_returns).mean(),
            "average_rmse": np.asarray(met.get_episode_rmse).mean(),
            "rmse_std": np.asarray(met.get_episode_rmse).std(),
            "worst_case_rmse_at_0.5": compute_cvar(np.asarray(met.get_episode_rmse), 0.5, lower_range=False),
            "failure_rate":  np.asarray(met.get_episode_constraint_violations).mean(),       
            "average_constraint_violation": np.asarray(met.get_episode_constraint_violation_steps).mean(),
            # others ???
        }
        return results
    
    

class RecordDataWrapper(gym.Wrapper):
    """A wrapper to standardizes logging for benchmark envs.
    
    currently saved info
    * obs, rew, done, info, act
    * env.state, env.current_preprocessed_action
    
    """
    def __init__(self, env, deque_size=None, **kwargs):
        super().__init__(env)
        self.deque_size = deque_size
        self.initialize_data_containers()

    def initialize_data_containers(self):
        """"""
        self.episode_data = defaultdict(list)
        self.data = defaultdict(lambda: deque(self.deque_size))
        
    def reset(self):
        """"""
        self.episode_data = defaultdict(list)
        if self.env.INFO_IN_RESET:
            obs, info = self.env.reset()
            step_data = dict(
                obs=obs, info=info, state=self.env.state
            )
            for key, val in step_data.items():
                self.episode_data[key].append(val)
            return obs, info
        else:
            obs = self.env.reset()
            step_data = dict(
                obs=obs, state=self.env.state
            )
            for key, val in step_data.items():
                self.episode_data[key].append(val)
            return obs 
    
    def step(self, act):
        """"""
        obs, rew, done, info = self.env.step(act)
        # save to episode data container
        step_data = dict(
            obs=obs, act=act, done=float(done), info=info, rew=rew, length=1,
            state=self.env.state, current_preprocessed_action=self.env.current_preprocessed_action
        )
        for key, val in step_data.items():
            self.episode_data[key].append(val)
        # save to data container
        if done:
            for key, ep_val in self.episode_data.items():
                self.data[key].append(deepcopy(ep_val))
            # re-initialize episode data container 
            self.episode_data = defaultdict(list)
        return obs, rew, done, info
    
    
    
class MetricExtractor:
    """A utility class that computes metrics given collected trajectory data.
    
    metrics that can be derived
    * episode lengths, episode total rewards/returns
    * RMSE (given the square error/mse is saved in info dict at each step)
    * episode occurrences of constraint violation 
        (0/1 for each episode, failure rate = #occurrences/#episodes)
    * episode constraint violation steps
        (how many constraint violations happened in each episode)

    """
    def __init__(self, data, **kwargs):
        self.data = data 
    
    def get_episode_data(self, key, postprocess_func=lambda x: x):
        """Extract data field from recorded trajectory data, optionally postprocess each episode data (e.g. get sum)."""
        if key in self.data:
            episode_data = [postprocess_func(ep_val) for ep_val in self.data[k]]
        elif key in self.data["info"][0][-1]:
            # if the data field is contained in step info dict
            episode_data = [postprocess_func([info.get(key, 0.) for info in ep_info]) 
                            for ep_info in self.data["info"]]
        else:
            raise KeyError("Given data key does not exist in recorded trajectory data.")
        return episode_data
    
    def get_episode_lengths(self):
        """Total length of episodes."""
        return self.get_episode_data("length", postprocess_func=sum)
    
    def get_episode_returns(self):
        """Total reward/return of episodes."""
        return self.get_episode_data("rew", postprocess_func=sum)

    def get_episode_rmse(self):
        """Root mean square error of episodes."""
        return self.get_episode_data("mse", 
                                     postprocess_func=lambda x: np.sqrt(np.mean(x)))

    def get_episode_constraint_violations(self):
        """Occurence of any violation in episodes."""
        return self.get_episode_data("constraint_violation", 
                                     postprocess_func=lambda x: float(any(x)))

    def get_episode_constraint_violation_steps(self):
        """Total violation steps of episodes."""
        return self.get_episode_data("constraint_violation", 
                                     postprocess_func=sum)
        

    
    
