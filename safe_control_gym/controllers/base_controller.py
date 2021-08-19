"""Base classes."""


class BaseController:
    """Template for controller/agent, implement the following methods as needed."""

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path="temp/model_latest.pt",
                 output_dir="temp",
                 device="cpu",
                 seed=0,
                 **kwargs):
        """Initializes controller agent.

        Args:
            env_func (callable): function to instantiate task/env.
            training (bool): training flag.
            checkpoint_path (str): file to save trained model & experiment state.
            output_dir (str): folder to write outputs.
            device (str): cpu or cuda.
            seed (int): random seed.
        """
        # base args
        self.env_func = env_func
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = device
        self.seed = seed

        # algo specific args
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def reset(self):
        """Do initializations for training or evaluation."""
        pass

    def close(self):
        """Shuts down and cleans up lingering resources."""
        pass

    def save(self, path):
        """Saves model params and experiment state to checkpoint path."""
        pass

    def load(self, path):
        """Restores model and experiment given checkpoint path."""
        pass

    def learn(self, env=None, **kwargs):
        """Performs learning (pre-training, training, fine-tuning, etc)."""
        pass

    def run(self,
            env=None,
            render=False,
            n_episodes=10,
            verbose=False,
            **kwargs):
        """Runs evaluation with current policy."""
        pass
