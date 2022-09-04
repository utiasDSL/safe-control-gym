# IROS 2022 Safe Robot Learning Competition

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/dgni-58OSMs/maxresdefault.jpg)](https://www.youtube.com/watch?v=dgni-58OSMs)

*(The image above links to the video of a flight example)*

*Note: beta release subject to change throughout the month of August 2022; register for updates*

- [Official Webpage](https://www.dynsyslab.org/iros-2022-safe-robot-learning-competition/)
- [IROS Competition Page](https://iros2022.org/program/competition/#toggle-id-8)
- [GitHub Discussions](https://github.com/utiasDSL/safe-control-gym/discussions/categories/iros-2022-competition)
- [Google Form](https://forms.gle/vEmVK99n1SyaE4Zw9) (to register your interest and receive e-mail updates)

## Description 

The task is to design a controller/planner that enables a quadrotor (*Crazyflie 2.x*) to **safely fly through a set of gates and reach a predefined target despite uncertainties in the robot dynamics (e.g., mass and inertia) and the environment (e.g., wind and position of the gates)**. The algorithms will be evaluated regarding their safety (e.g., no collisions) and performance (e.g., time to target). We encourage participants to explore both control and reinforcement learning approaches (e.g., robust, adaptive, predictive, learning-based and optimal control, and model-based/model-free reinforcement learning). The controller/planner has access to the position and attitude measurements provided by a motion capture system and the noisy pose of the closest next gate. The controller can [send position, velocity, acceleration and heading references to an onboard position controller](https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState).

## Install on Ubuntu/macOS

We recommend Ubuntu 20.04 on a mid-tier laptop and GPU (e.g., a Lenovo P52 with i7-8850H/Quadro P2000)

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
git checkout beta-iros-competition
```

Create and access a Python 3.8 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n safe python=3.8
conda activate safe
```

Install the `safe-control-gym` repository 

```bash
pip install --upgrade pip
pip install -e .
```

## Install `pycffirmware`

```bash
cd ..
git clone https://github.com/utiasDSL/pycffirmware.git
cd pycffirmware/
git submodule update --init --recursive
```

### On Ubuntu

```bash
sudo apt update
sudo apt -y install swig
sudo apt install build-essential
cd wrapper/
chmod +x build_linux.sh
conda activate safe
./build_linux.sh
```

### On macOS

Install [`brew`](https://brew.sh/), then
```bash
brew install swig
brew install gcc            # Also run `xcode-select --install` if prompted
brew install make
cd wrapper/
chmod +x build_osx.sh       # Assumes `gcc` is at `/usr/local/bin/gcc-12`
conda activate safe
./build_osx.sh
```

Also see how to install [SWIG](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/) and [`pycffirmware`](https://github.com/utiasDSL/pycffirmware)'s `README.md`

## Getting Started
Run the scripts in [`competition/`](https://github.com/utiasDSL/safe-control-gym/tree/beta-iros-competition/competition)
```bash
cd ../../safe-control-gym/competition/
python3 getting_started.py --overrides ./getting_started.yaml
```
**Modify file [`edit_this.py`](https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/edit_this.py) to customize your controller based on [Crazyswarm's Crazyflie interface](https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie)**

## Development and Evaluation Scenarios

A complete problem is specified by a YAML file, e.g. [`getting_started.yaml`](https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/getting_started.yaml)

Proposed solutions will be evaluated in 5 scenarios with different challenges:

| Evaluation Scenario | Constraints  | Rand. Inertial Properties | Randomized Obstacles, Gates | Rand. Between Episodes | Notes |
| :-----------------: | :----------: | :-----------------------: | :-------------------------: | :--------------------: | :---: |
| [`level0.yaml`][link0] | **Yes** | *No* | *No* | *No* | Perfect knowledge |
| [`level1.yaml`][link1] | **Yes** | **Yes** | *No* | *No* | Adaptive |
| [`level2.yaml`][link2] | **Yes** | **Yes** | **Yes** | *No* | Learning, re-planning |
| [`level3.yaml`][link3] | **Yes** | **Yes** | **Yes** | **Yes** | Robustness |
| | | | | | |
| sim2real | **Yes** | Real-life hardware |  **Yes**, injected | *No* | Sim2real transfer |

> "Rand. Between Episodes" (governed by argument `reseed_on_reset`) states whether randomized properties and positions vary or are kept constant (by re-seeding the random number generator on each `env.reset()`) across episodes
> 
> Note 1: the random seed used to score solution will be picked at the time of the competition
> 
> Note 2: if the base scenarios do not allow to determine a unique winner, we will progressively raise the difficulty by, alternately, (i) adding intermediate gates and (ii) increasing the parameters of the random distributions by 50% (except in `level0`).

[link0]: https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/level0.yaml
[link1]: https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/level1.yaml
[link2]: https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/level2.yaml
[link3]: https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/level3.yaml


## Implement Your Controller/Solution

Methods to Re-implement in [`edit_this.py`](https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/edit_this.py)

**Required**
```
edit_this.py : Controller.__init__(initial_obs, initial_info)           # Initialize the controller

    Args:
        initial_obs (ndarray): The initial observation of the quadrotor's state
            [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].

        initial_info (dict): The a priori problem information as a dictionary with keys

            - 'ctrl_timestep'                   Control time step (in seconds)
            - 'ctrl_freq'                       Control frequency (in Hz)
            - 'episode_len_sec'                 Maximum duration of an episode (in seconds)
            - 'nominal_physical_parameters'     *Nominal* mass and inertia of the quadrotor

            - 'gate_dimensions'                 Shape and measurements of the gates
            - 'obstacle_dimensions'             Shape and measurements of the obstacles
            - 'nominal_gates_pos_and_type'      *Nominal* pose and type (tall, low, etc.) of the gates
            - 'nominal_obstacles_pos'           *Nominal* pose of the obstacles
            - 'x_reference'                     Final position to reach/hover at

            - 'initial_state_randomization'     Distributions of the randomized additive error on the initial pose
            - 'inertial_prop_randomization'     Distributions of the randomized additive error on the inertial properties
            - 'gates_and_obs_randomization'     Distributions of the randomized additive error on the gates and obstacles positions
            - 'disturbances'                    Distributions of the dynamics and input disturbances  

            - 'symbolic_model'                  CasADi's 3D quadrotor dynamics
            - 'symbolic_constraints'            CasADi's constraints

    Returns: N/A
```

**Required**
```
edit_this.py : Controller.cmdFirmware(time, obs, reward, done, info)    # Select the next command for the quadrotor

    Args:
        time (float): Episode's elapsed time, in seconds.
        obs (ndarray): The quadrotor's pose from PyBullet or Vicon
            [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].

        reward (float, optional): The reward signal.
        done (bool, optional): Wether the episode has terminated.
        info (dict, optional): Current step information as a dictionary with keys

            - 'current_target_gate_id'          ID of the next gate (-1 when all gates have been traveled through)
            - 'current_target_gate_type'        Type of the next gate (0: tall, 1: low)
            - 'current_target_gate_in_range'    Boolean, whether the next gate is close enough (i.e., <= VISIBILITY_RANGE == 0.45m) for perfect visibility (affects the value of the next key 'current_target_gate_pos')
            - 'current_target_gate_pos'         *Nominal* or exact position of the next gate (depending on the value of the key above, 'current_target_gate_in_range')
            
            - 'at_goal_position'                Boolean, whether the quadrotor is at the final position ('x_reference')
            - 'task_completed'                  Boolean, whether the quadrotor stayed at the final position ('x_reference') for 2''

            - 'constraint_values'               Constraint evaluations
            - 'constraint_violation'            Boolean, whether any of the constraints is violated
            - 'collision'                       Collision, as a tuple (collided object id, boolean), note when False, ID==None

    Returns:
        Command: selected type of command (FINISHED, NONE, FULLSTATE, TAKEOFF, LAND, STOP, GOTO, NOTIFYSETPOINTSTOP, see Enum-like class `Command`).
        List: arguments for the type of command
            - FINISHED's args: []
            - NONE's args: []
            - FULLSTATE's args: [pos (3 val), vel (3 val), acc (3 val), yaw, rpy_rates (3 val), curr_time] 
            - TAKEOFF's args: [height, duration]
            - LAND's args: [height, duration]
            - STOP's args: []
            - GOTO's args: [x, y, z, yaw, duration, relative (bool)]
            - NOTIFYSETPOINTSTOP's args: []

        Also see: https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/competition_utils.py#L10
        and: https://crazyswarm.readthedocs.io/en/latest/api.html#crazyflie-class
```

**Optional, recommended for learning, adaptive control**
```
edit_this.py : Controller.interStepLearn(...)       # Update the controller's internal state at each step

    Args:
        N/A

    Leverage the data in `self.action_buffer`, `self.obs_buffer`, `self.reward_buffer`, self.done_buffer`, `self.info_buffer`

    Returns: N/A     
```

**Optional, recommended for learning, adaptive control**
```
edit_this.py : Controller.interEpisodeLearn(...)    # Update the controller's internal state between episodes

    Args:
        N/A

    Leverage the data in `self.action_buffer`, `self.obs_buffer`, `self.reward_buffer`, self.done_buffer`, `self.info_buffer`

    Returns: N/A
```

## Submission

- Fork this repository ([help](https://docs.github.com/en/get-started/quickstart/fork-a-repo))
- Checkout this branch (`beta-iros-competition`)
- Implement your solution by modifying [`edit_this.py`](https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/edit_this.py)
- Create a Pull Request into `utiasDSL/safe-control-gym:beta-iros-competition` from your fork ([help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork))
- Tag @JacopoPan in the Pull Request's Conversation tab

## Scoring

Solutions will be evaluated and scored by:
- Performance: minimizing the time (in sec.) required to complete the task (flying through the gates)
- Safety: minimizing the no. of collisions with gates and obstacles as well as the number of constraint violations
- Robustness: maximizing the success rate (across episodes) in uncertain scenarios (i.e., [`level3.yaml`](https://github.com/utiasDSL/safe-control-gym/blob/beta-iros-competition/competition/level3.yaml))
- Data and compute efficiency: minimizing the no. of episodes and the learning time (in sec.) used to improve performance 

## Prizes
- 1st: TBA
- 2nd: TBA
- 3rd: TBA

## Organizers
- Angela Schoellig (University of Toronto, Vector Institute)
- Davide Scaramuzza (University of Zurich)
- Vijay Kumar (University of Pennsylvania)
- Nicholas Roy (Massachusetts Institute of Technology)
- Todd Murphey (Northwestern University)
- Sebastian Trimpe (RWTH Aachen University)
- Wolfgang Hönig (TU Berlin)
- Mark Muller (University of California Berkeley)
- Jose Martinez-Carranza (INAOE)
- SiQi Zhou (University of Toronto, Vector Institute)
- Melissa Greeff (University of Toronto, Vector Institute)
- Jacopo Panerati (University of Toronto, Vector Institute)
- Wenda Zhao (University of Toronto, Vector Institute)
- Spencer Teetaert (University of Toronto)
- Yunlong Song (University of Zurich)
- Leticia Oyuki Rojas Pérez (INAOE)
- Adam W. Hall (University of Toronto, Vector Institute)
- Justin Yuan (University of Toronto, Vector Institute)
- Lukas Brunke (University of Toronto, Vector Institute)
- Antonio Loquercio (UC Berkeley)

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute for Artificial Intelligence](https://github.com/VectorInstitute)
