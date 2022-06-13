# safe-control-gym




Physics-based CartPole and Quadrotor [Gym](https://gym.openai.com) environments (using [PyBullet](https://pybullet.org/wordpress/)) with symbolic *a priori* dynamics (using [CasADi](https://web.casadi.org)) for **learning-based control**, and model-free and model-based **reinforcement learning** (RL). 

These environments include (and evaluate) symbolic safety constraints and implement input, parameter, and dynamics disturbances to test the robustness and generalizability of control approaches. [[PDF]](https://arxiv.org/pdf/2108.06266.pdf)

<img src="figures/problem_illustration.jpg" alt="problem illustration" width="800">

```
@article{brunke2021safe,
         title={Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning}, 
         author={Lukas Brunke and Melissa Greeff and Adam W. Hall and Zhaocong Yuan and Siqi Zhou and Jacopo Panerati and Angela P. Schoellig},
         journal = {Annual Review of Control, Robotics, and Autonomous Systems},
         year={2021},
         url = {https://arxiv.org/abs/2108.06266}}
```




## Install on Ubuntu/macOS

### Clone repo

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
```

### Option A (recommended): using conda 

Create and access a Python 3.8 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n safe python=3.8.10
conda activate safe
```

Install the `safe-control-gym` repository 

```
pip install --upgrade pip
pip install -e .
```


### Option B: using venv and poetry

Create and access a Python 3.8 virtual environment using
[`pyenv`](https://github.com/pyenv/pyenv) and
[`venv`](https://docs.python.org/3/library/venv.html)

```bash
pyenv install 3.8.10
pyenv local 3.8.10
python3 -m venv safe
source safe/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
```

#### Note:
You may need to separately install `gmp`, a dependency of `pycddlib`:
 ```bash
conda install -c anaconda gmp
 ```
 or 
  ```bash
 sudo apt-get install libgmp-dev
 ```


## Architecture

Overview of [`safe-control-gym`](https://arxiv.org/abs/2109.06325)'s API:

<img src="figures/block.png" alt="block diagram" width="800">

```
@misc{yuan2021safecontrolgym,
      title={safe-control-gym: a Unified Benchmark Suite for Safe Learning-based Control and Reinforcement Learning}, 
      author={Zhaocong Yuan and Adam W. Hall and Siqi Zhou and Lukas Brunke and Melissa Greeff and Jacopo Panerati and Angela P. Schoellig},
      year={2021},
      eprint={2109.06325},
      archivePrefix={arXiv},
      primaryClass={cs.RO}}
```


## Configuration

<img src="figures/config.png" alt="config" width="800"> 

## Performance

We compare the sample efficiency of `safe-control-gym` with the original [OpenAI Cartpole][1] and [PyBullet Gym's Inverted Pendulum][2], as well as [`gym-pybullet-drones`][3].
We choose the default physic simulation integration step of each project.
We report performance results for open-loop, random action inputs.
Note that the Bullet engine frequency reported for `safe-control-gym` is typically much finer grained for improved fidelity.
`safe-control-gym` quadrotor environment is not as light-weight as [`gym-pybullet-drones`][3] but provides the same order of magnitude speed-up and several more safety features/symbolic models.

| Environment              | GUI    | Control Freq.  | PyBullet Freq.  | Constraints & Disturbances^       | Speed-Up^^      |
| :----------------------: | :----: | :------------: | :-------------: | :-------------------------------: | :-------------: |
| [Gym cartpole][1]        | True   | 50Hz           | N/A             | No                                | 1.16x           |
| [InvPenPyBulletEnv][2]   | False  | 60Hz           | 60Hz            | No                                | 158.29x         |
| [cartpole][4]            | True   | 50Hz           | 50Hz            | No                                | 0.85x           |
| [cartpole][4]            | False  | 50Hz           | 1000Hz          | No                                | 24.73x          |
| [cartpole][4]            | False  | 50Hz           | 1000Hz          | Yes                               | 22.39x          |
| | | | | | |
| [gym-pyb-drones][3]      | True   | 48Hz           | 240Hz           | No                                | 2.43x           |
| [gym-pyb-drones][3]      | False  | 50Hz           | 1000Hz          | No                                | 21.50x          |
| [quadrotor][5]           | True   | 60Hz           | 240Hz           | No                                | 0.74x           |
| [quadrotor][5]           | False  | 50Hz           | 1000Hz          | No                                | 9.28x           |
| [quadrotor][5]           | False  | 50Hz           | 1000Hz          | Yes                               | 7.62x           |

> ^ Whether the environment includes a default set of constraints and disturbances
> 
> ^^ Speed-up = Elapsed Simulation Time / Elapsed Wall Clock Time; on a 2.30GHz Quad-Core i7-1068NG7 with 32GB 3733MHz LPDDR4X; no GPU

[1]: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
[2]: https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/mujoco/envs/pendulum/inverted_pendulum_env.py
[3]: https://github.com/utiasDSL/gym-pybullet-drones

[4]: https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/gym_control/cartpole.py
[5]: https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py

## Getting Started
Familiarize with APIs and environments with the scripts in [`examples/`](https://github.com/utiasDSL/safe-control-gym/tree/main/examples)
```
$ cd ./examples/                                                                    # Navigate to the examples folder
$ python3 tracking.py --overrides ./tracking.yaml                                   # PID trajectory tracking with the 2D quadcopter
$ python3 verbose_api.py --system cartpole --overrides verbose_api.yaml             #  Printout of the extended safe-control-gym APIs
```




## Systems Variables and 2D Quadrotor Lemniscate Trajectory Tracking

<img src="figures/systems.png" alt="systems" width="450"> <img src="figures/figure8.gif" alt="trajectory" width="350">


## Verbose API Example

<img src="figures/prints.png" al="prints" width="800">




## List of Implemented Controllers

- [LQR](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/lqr/lqr.py)
- [iLQR](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/lqr/ilqr.py)
- [Linear MPC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/linear_mpc.py)
- [GP-MPC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py)
- [SAC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/sac/sac.py)
- [PPO](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/ppo/ppo.py)
- [Safety Layer](https://github.com/utiasDSL/safe-control-gym/tree/main/safe_control_gym/controllers/safe_explorer)
- [RARL](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/rarl/rarl.py)
- [RAP](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/rarl/rap.py)
- [MPSC](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpsc/mpsc.py)
- [CBF](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/cbf/cbf_qp.py)




## Re-create the Results in "Safe Learning in Robotics" [[arXiv link]](https://arxiv.org/pdf/2108.06266.pdf)

To stay in touch, get involved or ask questions, please open an [issue on GitHub](https://github.com/utiasDSL/safe-control-gym/issues) or contact us via e-mail (`{jacopo.panerati, zhaocong.yuan, adam.hall, siqi.zhou, lukas.brunke, melissa.greeff}@robotics.utias.utoronto.ca`).


### Figure 6—Robust GP-MPC [[1]](https://ieeexplore.ieee.org/document/8909368)

```
$ cd ../experiments/annual_reviews/figure6/                        # Navigate to the experiment folder
$ chmod +x create_fig6.sh                                          # Make the script executable, if needed
$ ./create_fig6.sh                                                 # Run the script (ca. 2')
```
This will use the models in `safe-control-gym/experiments/figure6/trained_gp_model/` to generate

<img src="figures/gp-mpc.png" alt="gp-mpc" width="800">

To also re-train the GP models from scratch (ca. 30' on a laptop)
```
$ chmod +x create_trained_gp_model.sh                              # Make the script executable, if needed
$ ./create_trained_gp_model.sh                                     # Run the script (ca. 30')
```
> **Note:** this will backup and overwrite `safe-control-gym/experiments/figure6/trained_gp_model/`


-----

### Figure 7—Safe RL Exploration [[2]](https://arxiv.org/abs/1801.08757)

```
$ cd ../figure7/                                                   # Navigate to the experiment folder
$ chmod +x create_fig7.sh                                          # Make the script executable, if needed
$ ./create_fig7.sh                                                 # Run the script (ca. 5'')
```
This will use the data in `safe-control-gym/experiments/figure7/safe_exp_results.zip/` to generate

<img src="figures/safe-exp.png" alt="safe-exp" width="800">

To also re-train all the controllers/agents (**warning:** >24hrs on a laptop, if necessary, run each one of the loops in the Bash script—PPO, PPO with reward shaping, and the Safe Explorer—separately)
```
$ chmod +x create_safe_exp_results.sh                              # Make the script executable, if needed
$ ./create_safe_exp_results.sh                                     # Run the script (>24hrs)
```
> **Note:** this script will (over)write the results in `safe-control-gym/experiments/figure7/safe_exp_results/`; if you do not run the re-training to completion, delete the partial results `rm -r -f ./safe_exp_results/` before running `./create_fig7.sh` again.


-----

### Figure 8—Model Predictive Safety Certification [[3]](https://ieeexplore.ieee.org/document/8619829)

(required) Obtain [MOSEK's license](https://www.mosek.com/products/academic-licenses/) (free for academia).
Once you have received (via e-mail) and downloaded the license to your own `~/Downloads` folder, install it by executing
```
$ mkdir ~/mosek                                                    # Create MOSEK license folder in your home '~'
$ mv ~/Downloads/mosek.lic ~/mosek/                                # Copy the downloaded MOSEK license to '~/mosek/'
```
Then run
```
$ cd ../figure8/                                                   # Navigate to the experiment folder
$ chmod +x create_fig8.sh                                          # Make the script executable, if needed
$ ./create_fig8.sh                                                 # Run the script (ca. 1')
```
This will use the unsafe (pre-trained) PPO controller/agent in folder `safe-control-gym/experiments/figure8/unsafe_ppo_model/` to generate

<img src="figures/mpsc-1.png" alt="mpsc-1" width="800"> 

<img src="figures/mpsc-2.png" alt="mpsc-2" width="400"> <img src="figures/mpsc-3.png" alt="mpsc-3" width="400">

To also re-train the unsafe PPO controller/agent (ca. 2' on a laptop) 
```
$ chmod +x create_unsafe_ppo_model.sh                              # Make the script executable, if needed
$ ./create_unsafe_ppo_model.sh                                     # Run the script (ca. 2')
```
> **Note:** this script will (over)write the model in `safe-control-gym/experiments/figure8/unsafe_ppo_model/`




# References
- [1] Hewing L, Kabzan J, Zeilinger MN. 2020. [Cautious model predictive control using Gaussian process regression](https://ieeexplore.ieee.org/document/8909368). IEEE Transactions on Control Systems Technology 28:2736–2743
- [2] Dalal G, Dvijotham K, Vecerik M, Hester T, Paduraru C, Tassa Y. 2018. [Safe exploration in continuous action spaces](https://arxiv.org/abs/1801.08757). arXiv:1801.08757 [cs.AI]
- [3] Wabersich KP, Zeilinger MN. 2018. [Linear Model Predictive Safety Certification for Learning-Based Control](https://ieeexplore.ieee.org/document/8619829). In 2018 IEEE Conference on Decision and Control (CDC), pp. 7130–7135




# Related Open-source Projects
- [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones): single and multi-quadrotor environments
- [`gym-marl-reconnaissance`](https://github.com/JacopoPan/gym-marl-reconnaissance): multi-agent heterogeneous (UAV/UGV) environments
- [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3): PyTorch reinforcement learning algorithms
- [`bullet3`](https://github.com/bulletphysics/bullet3): multi-physics simulation engine
- [`gym`](https://github.com/openai/gym): OpenAI reinforcement learning toolkit
- [`safety-gym`](https://github.com/openai/safety-gym): environments for safe exploration in RL
- [`realworldrl_suite`](https://github.com/google-research/realworldrl_suite): real-world RL challenge framework
- [`casadi`](https://github.com/casadi/casadi): symbolic framework for numeric optimization



# Desiderata/WIP/Summer 2022 Internships TODOs
- Publish to [PyPI](https://realpython.com/pypi-publish-python-package/)
- [Colab](https://colab.research.google.com/notebooks/intro.ipynb) examples
- Create a list of FAQs from [Issues tagged as questions](https://github.com/utiasDSL/safe-control-gym/issues?q=is%3Aissue+is%3Aopen+label%3Aquestion)
- Link [papers](https://www.semanticscholar.org/paper/safe-control-gym%3A-a-Unified-Benchmark-Suite-for-and-Yuan-Hall/66b4656ab7732dcdcf39c466e8ab948c2b4a042d#citingPapers), projects, blog posts (Cat's, etc.) using `safe-control-gym`

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute for Artificial Intelligence](https://github.com/VectorInstitute)
