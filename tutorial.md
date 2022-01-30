 # Safe Control Gym - How to Get Started 

To run an experiment in safe-control-gym, there are elements required: 

1. A control approach - implementations of control approaches are included in repo (see the Control Approaches section for a list and description) or implement your own controller
2. A robotic model - choose a robotic model or implement your own
    - cartpole
    - 1D quadrotor
    - 2D quadrotor
3. A configuration file - this provides all the relevant information for what is actually happening in your experiement (more on this later)

Execute the following command to run a basic training loop with a PPO controller on the cartpole system. 

```
cd experiments
python3 ../../main.py --algo ppo --task cartpole --overrides $CONFIG_PATH --output_dir ${OUTPUT_DIR} --tag $TAG_ROOT/$TAG --thread $thread --seed $seed
```
<!-- Note: need to change this to actual values  -->

Let's go through an example in code to understand how these parts come together. This example is pulled from main.py in XXX. 

### Step 1 - Load in your configuration

The configuration file is handled through the class `ConfigFactory()`. To add additional configuration parameters through the command line use the `add_argument()` method

```
config = ConfigFactory()
config.add_argument("--func", type=str, default="train", help="main function to run.")
```
### Step 2 - Execute experiment

For general cases, this can be done through calling any function that executes the desired task. For this example, we will show a training loop but note that main.py has the option to select the function to execute through a commandline argument. 

```
train(config)
```
#### Initializing an environment and controller 
If we consider the function for training as an example, first we define the function to create the environment 
```
 env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
```

Then initialize the controller and control agent 
```
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                         output_dir=config.output_dir,
                         device=config.device,
                         seed=config.seed,
                         **config.algo_config)
```

#### Training an agent 

Training can be executed in one line given the used controller uses the BaseController class for guidance as follows 

```
control_agent.learn()
control_agent.close()
```

#### Testing an agent

To test a trained model, load the model as follows 

```
control_agent.load(os.path.join(config.restore, "model_latest.pt"))
```

Then, run the agent to test the performance of the controller

```
results = control_agent.run(n_episodes=config.algo_config.eval_batch_size,
                            render=config.render,
                            verbose=config.verbose,
                            use_adv=config.use_adv)
```

After testing, the results can be extracted in different ways. 

Configuring training, testing, and controller parameters will be discussed in the next section. 

### Using overrides/configuration files 

Each controller has a default configuration file with bare-bones parameters implemented. These are stored in the controllers directory with the implementations of the controllers. To add to and modify the exisiting configurations, the override commandline option can be used. 

Here is a condensed list of the options to set in the configuration file. For controller specific and additional options, refer to yaml files in the examples and experiment directories. 

##### A note on configuration management using `Registry()`:

The `Registry()` class is a singleton that manages all the default configurations in safe-control-gym. The registry uses an id to fetch the required configuration file. The assignments of the ids to their respective yaml files can be found in __init__.py for controllers and __init__.py for envs (or tasks)


#### Command Line Options

The configuration used by the experiment is specified in three ways via the command line 
1. --algo : this loads the default configuration for controller/algorithm used in the experiment
2. --task : this loads the default configuration associated with the agent model used in the experiment 
3. --overrides : this loads an overrides file that makes an desired modifications to the default configuration files you are using 


#### Experiment Configurations 

- task_config: specifies the kind of 
    - used in override to signal start of the task configuration parameters
- algo_config: 
    - used in override to signal start of the controller configuration parameters 
    - defined in __init__.py in XX (controllers)
- env_config:
    - Specify the cost/reward function, constraints, disturbances
- kinds of tasks - stabilization and traj tracking 
- algo and task load the basic configs - use overrides to modify the parameters

### Other command line options 
--tag
--seed
--device
--output_dir
--restore

#### Control Approaches 
1. Control and Safe Control Baselines:
    - LQR 
    - iLQR 
    - LMPC 
    - NMPC
2. Reinforcement Learning Baselines:
    - PPO (Proximal Policy Optimization)
    - SAC (Soft-Actor Critic)
3. Safe Learning-based Control 
    - GP-MPC
4. Safe and Robust Reinforcement Learning 
    - RARL 
    - RAP
5. Safety Ceritification of Learned Controllers 
    - MPSC 
    - CBF 
6. Safe Exploration 
    - Safe PPO





