## Adding a New Controller
- Create a new folder in `safe-control-gym/controllers` with the name of your controller
- Add a blank `__init__.py` file, a file for the base code of your controller called `<CONTROLLER_NAME>.py`, and a YAML file for the standard configuration for your controller called `<CONTROLLER_NAME>.yaml`. 
- In `<CONTROLLER_NAME>.py` extend the `BaseController` class found in `safe-control-gym/controllers/base_controller.py` to implement your controller. If necessary, create extra utility files called `<CONTROLLER_NAME>_utils.py` in that same folder.
- In the YAML file, put in the basic configuration needed for your controller, which will be the variables in the instantiation of your controller object.
- In `safe-control-gym/controllers/__init__.py`, register your new controller by adding a new registration command.

See `safe_control_gym/controllers/pid/` for a simple example.

## Creating an Experiment
- In `experiements/`, create a new folder with the name of your controller. 
- Add a blank `<CONTROLLER_NAME>_experiment.sh` file for running your experiment, a python file for the experiment called `<CONTROLLER_NAME>_experiment.py`, and a YAML file for the configuration for your control task called `config_<CONTROLLER_NAME>_<TASK>.yaml`, where `<TASK>` is the task in the experiment (either `quadrotor` or `cartpole`).
- In `<CONTROLLER_NAME>_experiment.sh`, enter the command-line script to run the experiment:

```
#!/bin/bash

# <CONTROLLER_NAME> Experiment.
python3 ./<CONTROLLER_NAME>_experiment.py --task <TASK> --algo <CONTROLLER_NAME> --overrides ./config_<CONTROLLER_NAME>_<TASK>.yaml
```
- Run `chmod -x <CONTROLLER_NAME>_experiment.sh` to make the file executable.
- In the YAML file, put in the configuration needed for the control task. Put the variables in a YAML object called `<TASK>_config`. 
- In `<CONTROLLER_NAME>_experiment.py`, create the code to run the experiment. 
    - Gather the configurations using: 
    ```
    CONFIG_FACTORY = ConfigFactory()               
    config = CONFIG_FACTORY.merge()
    ```
    - Instantiate the controller using:
    ```
    env_func = partial(make,
                    config.task,
                    **config.<TASK>_config
                    )
    ctrl = make(config.algo,
                env_func,
                )
    ```
    - Run the controller using:
    ```
    results = ctrl.run(<ADDITIONAL_ARGUMENTS>)
    ```
    - Display the results in any way you like
- Run the new controller py running `./<CONTROLLER_NAME>_experiment.sh`.

See `experiments/pid/` for a simple example.
