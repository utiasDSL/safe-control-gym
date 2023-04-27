# Safe Control Gym - How to Get Started

Refer to [this paper](https://arxiv.org/abs/2109.06325) for information on the motivation behind this gym as well as details on the implementation.

## Running a basic experiment

To run an experiment in safe-control-gym, the elements required are:

1. A control approach or algorithm - choose an existing implementation of a control approaches (see the Control Approaches section for a list) or implement your own controller
2. A robotic model - choose a robotic model (cartpole, 2D or 1D quadrotor) or implement your own
3. (Optionally) A safety filter - choose an existing implementation of a safety filter to guarantee the safety of system (see the Safety Filters section for a list) or implement your own
4. Configuration files - these provide all the relevant information for the control approach, environment, and task

See the examples in the `examples/` folder for example experiments using different controllers, environments, tasks, and safety filters. Let us see the necessary steps below:

### Step 1: Create the folder and files
In `experiments/`, create a new folder for your experiment. Then, add a blank `<EXPERIMENT_NAME>.sh` file for running your experiment, a python file for the experiment called `<EXPERIMENT_NAME>.py`, and a folder called `config_overrides` for all the YAML override files you need.

### Step 2: Create required configuraton files
The environment (which includes the task) and the controller must be configured. These may be for various experiments that you wish to conduct, and can all be collected in the `config_overrides` folder of your experiment. See `examples/lqr/config_overrides` for an example.

### Step 3: Create basic shell script
Fill out `<EXPERIMENT_NAME>.sh`, the simple shell script defining the command to run the experiment. An example can be found in `examples/lqr/lqr_experiment.sh`. This script should, at minimum, follow the structure:
```bash
#!/bin/bash

# <EXPERIMENT_NAME>.
python3 ${EXPERIMENT_NAME}.py \
    --task ${SYSTEM_NAME} \
    --algo ${CONTROLLER} \
    --overrides \
        ./config_overrides/${RELATIVE_PATH_TO_OVERRIDE_1} \
        ./config_overrides/${RELATIVE_PATH_TO_OVERRIDE_2} \
        ...
```

### Step 4: Merge configuration
Now we can write the python experiment file, `<EXPERIMENT_NAME>.py`. In the main function, we will start by merging the config that has been defined using the system, controller, and overrides in the shell script.
```python
    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
```

### Step 5: Create environment and controller
Using the `config` dictionary, we can create the environment and controller.
```python
    # Create an environment
    env_func = partial(make,
                    config.task,
                    **config.task_config
                    )

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
```

### Step 6: Train the controller
If the controller requires training, or loading of a pre-trained model, this can be done now. Each controller is trained differently, so consult the source code for the controller.


### Step 7: Run the experiment
Using the `BaseExperiment` class, we can run our desired experiment.
```python
    # Run the experiment.
    experiment = BaseExperiment(ctrl.env, ctrl)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=1)
    experiment.close()
```
This will return standard data on the trajectory, and metrics calculated over that experiment. Consult the `BaseExperiment` class for more details. The data from the experiment can be displayed or saved in any way you like. To run the experiment, simply run `./<EXPERIMENT_NAME>.sh` in the terminal.

## Using configuration/override files

Each controller has a default configuration file with the bare-bones parameters initialized. These are stored in `safe-control-gym/safe_control_gym/controllers` with the implementations of the controllers. Here is a good place to start if you want to know more about how to start working with a controller. To add to and modify the existing configuration, an override file is used. These configurations are merged together using the `.merge()` method.

### Command Line Options

The configuration used by the experiment is specified via the command line:

1. --algo : this specifies the default configuration for controller/algorithm used in the experiment
2. --task : this specifies the default configuration associated with the agent model used in the experiment
3. --safety_filter : this specifies the default configuration associated with the safety filter used in the experiment, if any are used.
4. --overrides : this specifies an overrides file that makes an desired modifications to the default configuration files you are using
5. --kv_overrides : this allows overriding specific values in the config using their key
6. --your_arg : using the `.add_argument()` method from `ConfigFactory()` in code and commandline arguments

### Override Configurations

For an experiment, you will need to change and specify some parameters using the default as a starting point. To do this, we specify the algorithm and task configurations in the override file loaded via the command line.

Here is one way of setting up your overrides file. Open up `safe-control-gym/examples/lqr/config_overrides/` for an example.

- task_config:
    - Contains all task override configurations including:
        - control frequency, constraints, disturbances, cost, initial state etc.
- algo_config:
    - Contains all algorithm configurations including:
        - seed, initial model parameters, number of iterations, learning rate etc.
- sf_config:
    - Contains all safety filter configurations

## Existing Control Approaches

### Control and Safe Control Baselines

| Approach | id | Location |
| -------- | --- | ----------- |
|  PID Controller    | 'pid' |   [PID](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/pid/pid.py)   |
|  Linear Quadratic Regulator    | 'lqr' |   coming soon   |
|  Iterative Linear Quadratic Regulator   | 'ilqr' |   coming soon  |

### Reinforcement Learning Baselines

| Approach | id | Location |
| -------- | --- | ----------- |
|  Proximal Policy Optimization | 'ppo' | [PPO](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/ppo/ppo.py) |
|  Soft-Actor Critic  | 'sac' | [SAC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/sac/sac.py) |

### Safe Learning-based Control
| Approach | id | Location |
| -------- | --- | ----------- |
|  Model Predictive Control w/ a Gaussian Process Model | 'gp_mpc' | [GP-MPC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py)  |
|  Linear Model Predictive Control | 'linear_mpc' | [Linear MPC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/linear_mpc.py) |

### Safe and Robust Reinforcement Learning
| Approach | id | Location |
| -------- | --- | ----------- |
|  Robust Adversarial Reinforcement Learning | 'rarl' | [RARL](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/rarl/rarl.py) |
|  Robust Adversarial Reinforcement Learning using Adversarial Populations | 'rap'  | [RAP](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/rarl/rap.py) |

### Safety Filters
| Approach | id | Location |
| -------- | --- | ----------- |
|  Model Predictive Safety Certification | 'linear_mpsc' | [MPSC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/safety_filters/mpsc/mpsc.py) |
|  Control Barrier Functions  | 'cbf'  |  [CBF](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/safety_filter/cbf/cbf.py) |
|  Neural Network Control Barrier Functions  | 'cbf_nn'  |  [CBF](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/safety_filter/cbf/cbf_nn.py) |

#### Safe Exploration
| Approach | id | Location |
| -------- | --- | ----------- |
|  Safety Layer | 'safe_explorer_ppo' |  [Safety Layer](https://github.com/utiasDSL/safe-control-gym/tree/main/safe_control_gym/controllers/safe_explorer)  |

## Adding a New Controller
- Create a new folder in `safe-control-gym/controllers` with the name of your controller
- Add a blank `__init__.py` file, a file for the base code of your controller called `<CONTROLLER_NAME>.py`, and a YAML file for the standard configuration for your controller called `<CONTROLLER_NAME>.yaml`.
- In `<CONTROLLER_NAME>.py` extend the `BaseController` class found in `safe-control-gym/controllers/base_controller.py` to implement your controller. If necessary, create extra utility files called `<CONTROLLER_NAME>_utils.py` in that same folder.
- In the YAML file, put in the basic configuration needed for your controller, which will be the variables in the instantiation of your controller object.
- In `safe-control-gym/controllers/__init__.py`, register your new controller by adding a new registration command.

See `safe_control_gym/controllers/pid/` for a simple example.
