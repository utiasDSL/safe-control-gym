# Reducing Maximization Bias and Risk in Hyperparameter Optimization for Reinforcement Learning and Learning-Based Control

This is the code for the paper entitiled `Reducing Maximization Bias and Risk in Hyperparameter Optimization for Reinforcement Learning and Learning-Based Control`. The implementation is adapted and based on [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym).

## Install on Ubuntu

### Create a `conda` environment

Create and access a Python 3.10 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n pr-env python=3.10
conda activate pr-env
```

### Install

```bash
pip install --upgrade pip
pip install -e .
```

#### Note

You may need to separately install `gmp`, a dependency of `pycddlib`:

 ```bash
conda install -c anaconda gmp
 ```

or

 ```bash
 sudo apt-get install libgmp-dev
 ```

 To perform hyperparmeter optimization, you may need `MySQL` database:

 ```bash
 sudo apt-get install mysql-server
 ```

 To set up, run the following commands sequencially:

 ```bash
sudo mysql
CREATE USER optuna@"%";
CREATE DATABASE {algo}_hpo;
GRANT ALL ON {algo}_hpo.* TO optuna@"%";
exit
 ```

 You may replace `{algo}` with `gp_mpc`, `ppo`, `sac`, or `ddpg` in order to run the scripts.

### Toy Examples

The results for toy examples in the paper can be reproduced in [`toy_example.ipynb`](experiments/comparisons/rl/toy_example.ipynb)


### Reinforcement Learning

To run hyperparameter optimization (HPO) for DDPG, run:

```bash
bash experiments/comparisons/rl/main.sh hostx TPESampler ddpg cartpole stab False
```

To run hyperparameter optimization (HPO) for PPO, run:

```bash
bash experiments/comparisons/rl/main.sh hostx TPESampler ppo cartpole stab False
```

To run hyperparameter optimization (HPO) for SAC, run:

```bash
bash experiments/comparisons/rl/main.sh hostx TPESampler sac cartpole stab False
```

### Learning-Based Control

To run hyperparameter optimization (HPO) for GP-MPC, run:

```bash
bash experiments/comparisons/gpmpc/main.sh hostx TPESampler cartpole stab False
```

#### Note
You may need to adjust the `path` of `conda.sh` in the sub-scripts called by `main.sh` such as `rl_hpo_strategy_eval.sh`.