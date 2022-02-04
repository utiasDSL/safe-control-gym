# Safe Control Gym - How to Get Started 

Refer to [this paper](https://arxiv.org/abs/2109.06325) for information on the motivation behind this gym as well as details on the implementation.

## Running a basic experiment

To run an experiment in safe-control-gym, the elements required are: 

1. A control approach or algorithm - choose an existing implementation of a control approaches (see the Control Approaches section for a list) or implement your own controller
2. A robotic model - choose a robotic model (cartpole, 2D or 1D quadrotor) or implement your own
3. A configuration file - this provides all the relevant information for what is actually happening in your experiment (more on this later)

The following command runs a basic training loop with a PPO controller on the cartpole system. This will take several minutes.

```
cd safe_control_gym/walkthroughs
python3 tutorial.py --algo ppo --task cartpole --overrides ./tutorial_configs/tutorial_ppo_cartpole.yaml --output_dir ./tutorial_models --tag tutorial_results/ppo --thread 1 --seed 222
```
If you have a GPU available, run the example with cuda:

```
python3 tutorial.py --algo ppo --task cartpole --overrides ./tutorial_configs/tutorial_ppo_cartpole.yaml --output_dir ./tutorial_models --tag tutorial_results/ppo --thread 1 --seed 222 --device cuda
```

Open up tutorial.py and let's look at how this example runs.

### Step 1 - Load in your configuration

```
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
```

The configuration file is handled through the class `ConfigFactory()`. To add additional configuration parameters through the command line, use the `add_argument()` method. We'll talk more about configuration files in the Experiment Configurations section. 

### Step 2 - Execute experiment

This basic script provides an example implementation of a training function, a testing function, and a plotting function. Here, we specify which portion of the experiment to run using the command line argument that was added above `--func' and the keys from the function dictionary.
```
MAIN_FUNCS = {"train": train, "plot": make_plots, "test": test_policy}
```
Each of the functions perform a different part of the experiment. 

```
func(config)
```

#### Initializing an environment and agent 

When training or testing, you will need to set up your agent and environment. To do this, first we define the function to create the environment which sets the task specifications and output locations. 
```
 env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
```
Then, initialize and reset
```
control_agent = make(config.algo,
                        env_func,
                        training=True,
                        checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                        output_dir=config.output_dir,
                        device=config.device,
                        seed=config.seed,
                        **config.algo_config)
control_agent.reset()
```
The make function takes an id (here config.algo), a set of arguments (here env_func, training etc.), and the config (here config.algo_config) and creates an instance of the callable (here control_agent) with all the specified configurations. `env_func` is used to load in the task configurations.

#### Training an agent 

Training can be executed in one line given the format of the `BaseController()` class as follows. To see more about how the base controller class is implemented or if you plan on creating your own controller, see `controllers/base_controller.py`

```
control_agent.learn()
control_agent.close()
```

#### Testing an agent

For existing controller's implemented using the `BaseController()` class, use the `.run()` method to test the performance of the controller

```
results = control_agent.run(n_episodes=config.algo_config.eval_batch_size,
                            render=config.render,
                            verbose=config.verbose,
                            use_adv=config.use_adv)
```

Note: If you want to load the model from a trained model, you can do as follows and use the `.load()` method. 
```
control_agent.load(os.path.join(config.restore, "model_latest.pt"))
```

After testing, the results can be extracted in different ways. 

#### Plotting results 

The `safe-control-gym` has plotting capabilities imported as `safe_control_gym.utils.plotting` that use the data saved to the output directory to use for visualization after running an experiment. To run plotting with this example, specify `--func plot` in the command line
```
python3 tutorial.py --func plot --tag tutorial_results/ppo --thread
```
This will execute `make_plots` in the example which does as follows
```
log_dir = os.path.join(config.output_dir, "logs")
plot_dir = os.path.join(config.output_dir, "plots")
mkdirs(plot_dir)
plot_from_logs(log_dir, plot_dir, window=3)
```
Here, log_dir is the location of the stored logs which is automatically the logs directory of your output directory when you train. The plot_dir is where you want to plots stored. `plot_from_logs` will generate a plot for each stat in the logs. If you only want to plot certain stats, you can specify a fourth argument, keys. 

## Using configuration/override files 

Each controller has a default configuration file with the bare-bones parameters initialized. These are stored in `safe-control-gym/safe_control_gym/controllers` with the implementations of the controllers. Here is a good place to start if you want to know more about how to start working with a controller. To add to and modify the existing configuration, an override file is used. These configurations are merged together using the `.merge()` method in your code, as shown in "Step 1 - Load in your configuration". 

#### A note on configuration management using `Registry()`:

The `Registry()` class is a singleton that manages all the default configurations in safe-control-gym. The registry uses an id to fetch the required configuration file. The assignments of the ids to their respective yaml files can be found in `safe-control-gym/safe_control_gym/controllers/__init__.py` for controllers and `safe-control-gym/safe_control_gym/envs/__init__.py` for envs (or tasks). Generally, it is the file name of where the controller or agent type is implemented. 

### Command Line Options

The configuration used by the experiment is specified in four ways via the command line:

1. --algo : this specifies the default configuration for controller/algorithm used in the experiment
2. --task : this specifies the default configuration associated with the agent model used in the experiment 
3. --overrides : this specifies an overrides file that makes an desired modifications to the default configuration files you are using 
4. --your_arg : using the `.add_argument()` method from `ConfigFactory()` in code and commandline arguments

### Override Configurations

For an experiment, you will need to change and specify some parameters using the default as a starting point. To do this, we specify the algorithm and task configurations in the override file loaded via the command line.

Here is one way of setting up your overrides file. Open up `safe-control-gym/walkthroughs/tutorial_configs/tutorial_ppo_cartpole.yaml` for an example.

- task_config: 
    - Contains all task override configurations including: 
        - control frequency, constraints, disturbances, cost, initial state etc. 
    - More specifics on task and environment configuration can be found in the "Environment Configuration" and "Task Configuration" sections
- algo_config: 
    - Contains all algorithm configurations including:
        - seed, initial model parameters, number of iterations, learning rate etc. 

## Configurations and Options 

### Other common arguments

These are some examples of common options you'd want to set in your experiment. Different control approaches may have different options you'd want to specify. For example, in the above code, `use_adv` can be specified for PPO to evaluate against an adversary. These can be specified via the command line or directly in your overrides file. 

| Arguments | Purpose | Use |
| ---- | ------ | ------ |
| tag | id of the experiment | N/A |
| seed | Randomization seed | `set_seed_from_config(config)` and input as arg at `make()`: `seed=config.seed` |
| device | Where to perform training ("cpu" or "cuda") | `set device_from_config(config)` and input as arg at `make()`: `device=config.device` | 
| output_dir | Where to stored output models | `set_dir_from_config(config)` and input as arg at `make()`: `output_dir=config.output_dir` |
| thread | How many threads to use | `torch.set_num_threads(config.thread)` | 

For more information on some common utilities in this repo, refer to `safe-control-gym/safe_control_gym/utils`

### Existing Control Approaches 

#### Control and Safe Control Baselines

<!-- | Approach | id | Location | 
| -------- | --- | ----------- |
|  LQR     | 'lqr' |          |
|  iLQR    | 'ilqr' |          | -->
<!-- - LQR: "lqr"- iLQR: "ilqr"- LMPC: "lmpc" - NMPC: "nmpc" -->

#### Reinforcement Learning Baselines 

| Approach | id | Location | 
| -------- | --- | ----------- |
|  Proximal Policy Optimization | 'ppo' | [PPO](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/ppo/ppo.py) |
|  Soft-Actor Critic  | 'sac' | [SAC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/sac/sac.py) |

#### Safe Learning-based Control
| Approach | id | Location | 
| -------- | --- | ----------- |
|  Model Predictive Control w/ a Gaussian Process Model | 'gp_mpc' | [GP-MPC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py)  |
|  Linear Model Predictive Control | 'gp_mpc' | [Linear MPC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/linear_mpc.py) |

#### Safe and Robust Reinforcement Learning
| Approach | id | Location | 
| -------- | --- | ----------- |
|  Robust Adversarial Reinforcement Learning | 'rarl' | [RARL](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/rarl/rarl.py) |
|  Robust Adversarial Reinforcement Learning using Adversarial Populations | 'rap'  | [RAP](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/rarl/rap.py) |

#### Safety Ceritification of Learned Controllers
| Approach | id | Location | 
| -------- | --- | ----------- |
|  Model Predictive Safety Certification | 'mpsc' | [MPSC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpsc/mpsc.py) |
<!-- |  CBF  | 'cbf'  |  Control Barrier Function - | -->

#### Safe Exploration 
| Approach | id | Location | 
| -------- | --- | ----------- |
|  Safety Layer | 'safe_explorer_ppo' |  [Safety Layer](https://github.com/utiasDSL/safe-control-gym/tree/main/safe_control_gym/controllers/safe_explorer)  |

#### Environment Configuration (Cost, Disturbance, Constraints)
### Cost Types
1. Quadratic
2. rl_reward

The implementations for the rewards depend on the agent you are using (cartpole or quadrotor). Please refer to `safe-control-gym/safe_control_gym/envs` for the classes that implement these tasks. 

#### Disturbance Types:
| Applied to | Types |
| ---- | ---- |
| action | white noise, impulse, step |
| observation | white noise, impulse, step |
| dynamics | white noise, impulse, step, adversary forces (deterministic) | 

Example: 
```
disturbances:
    observation:
        - disturbance_func: white_noise
          std: 0.05
```
<!-- #### Randomization

- Randomize initial state 
    distributions:
        - uniform 
        - choice -->

Constraint Types:
| Applied to | Types |
| ---- | ----- | 
| state | bounded_constraint, linear, quadratic, default_contraint | 
| input | bounded_constraint, linear, quadratic, default_contraint |

#### Benchmarking tasks 
These tasks are designed to be used in benchmarking experiments:
1. Stabilization - `stabilization`
2. Trajectory tracking - `traj_tracking`

The tasks have been implemented at the controller level.





