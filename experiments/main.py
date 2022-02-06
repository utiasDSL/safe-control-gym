"""Template training/plotting/testing script.

"""
import os
import sys
from functools import partial
import os
import pickle
import sys
import torch

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video


def train(config):
    """Training template.
    
    Usage:
        * to start training, use with `--func train`.
        * to restore from a previous training, additionally use `--restore {dir_path}` 
            where `dir_path` is the output folder from previous training.  

    """
    # Experiment setup.
    if not config.restore:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)
    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, "model_latest.pt"))
    # Training.
    control_agent.learn()
    control_agent.close()
    print("Training done.")


def make_plots(config):
    """Produces plots for logged stats during training.
    
    Usage
        * use with `--func plot` and `--restore {dir_path}` where `dir_path` is 
            the experiment folder containing the logs.
        * save figures under `dir_path/plots/`.

    """
    # Define source and target log locations.
    log_dir = os.path.join(config.output_dir, "logs")
    plot_dir = os.path.join(config.output_dir, "plots")
    mkdirs(plot_dir)
    plot_from_logs(log_dir, plot_dir, window=3)
    print("Plotting done.")


def test_policy(config):
    """Run the (trained) policy/controller for evaluation.
    
    Usage
        * use with `--func test`.
        * to test policy from a trained model checkpoint, additionally use 
            `--restore {dir_path}` where `dir_path` is folder to the trained model.
        * to test un-trained policy (e.g. non-learning based), use as it is.

    """
    # Evaluation setup.
    set_device_from_config(config)
    if config.set_test_seed:
        # seed the evaluation (both controller and env) if given
        set_seed_from_config(config)
        env_seed = config.seed
    else:
        env_seed = None
    # Define function to create task/env.
    env_func = partial(make, config.task, seed=env_seed, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    control_agent = make(config.algo,
                         env_func,
                         training=False,
                         checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()
    if config.restore:
        control_agent.load(os.path.join(config.restore, "model_latest.pt"))
    # Test controller.
    results = control_agent.run(n_episodes=config.algo_config.eval_batch_size,
                                render=config.render,
                                verbose=config.verbose,
                                use_adv=config.use_adv)
    # Save evalution results.
    if config.eval_output_dir is not None:
        eval_output_dir = config.eval_output_dir
    else:
        eval_output_dir = os.path.join(config.output_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)
    # test trajs and statistics 
    eval_path = os.path.join(eval_output_dir, config.eval_output_path)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "wb") as f:
        pickle.dump(results, f)
    ep_lengths = results["ep_lengths"]
    ep_returns = results["ep_returns"]
    mse = results["mse"]
    msg = "eval_ep_length {:.2f} +/- {:.2f}\n".format(ep_lengths.mean(), ep_lengths.std())
    msg += "eval_ep_return {:.3f} +/- {:.3f}\n".format(ep_returns.mean(), ep_returns.std())
    msg += "eval_mse {:.3f} +/- {:.3f}\n".format(mse.mean(), mse.std())
    print(msg)
    if "frames" in results:
        save_video(os.path.join(eval_output_dir, "video.gif"), results["frames"])
    control_agent.close()
    print("Evaluation done.")


MAIN_FUNCS = {"train": train, "plot": make_plots, "test": test_policy}


if __name__ == "__main__":
    # Make config.
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="train", help="main function to run.")
    fac.add_argument("--thread", type=int, default=0, help="number of threads to use (set by torch).")
    fac.add_argument("--render", action="store_true", help="if to render in policy test.")
    fac.add_argument("--verbose", action="store_true", help="if to print states & actions in policy test.")
    fac.add_argument("--use_adv", action="store_true", help="if to evaluate against adversary.")
    fac.add_argument("--set_test_seed", action="store_true", help="if to set seed when testing policy.")
    fac.add_argument("--eval_output_dir", type=str, help="folder path to save evaluation results.")
    fac.add_argument("--eval_output_path", type=str, default="test_results.pkl", help="file path to save evaluation results.")
    config = fac.merge()
    # System settings.
    if config.thread > 0:
        # E.g. set single thread for less context switching
        torch.set_num_threads(config.thread)

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception("Main function {} not supported.".format(config.func))
    func(config)


