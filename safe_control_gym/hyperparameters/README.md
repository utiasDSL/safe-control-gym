# Instruction for running hyperparameter optimization

## Install on Ubuntu

### Create a `conda` environment

Create the environment by following the steps presented in the main
[`README.md`](README.md)


To perform hyperparmeter optimization, you may need `MySQL` database (optional):

```bash
sudo apt-get install mysql-server # MySQL server starts automatically after the installation
```

To set up, run the following commands sequentially:

```bash
sudo mysql
CREATE USER optuna@"%"; # this creates the user name 'optuna' accessible by any host
CREATE DATABASE {algo}_hpo;
GRANT ALL ON {algo}_hpo.* TO optuna@"%";
exit
```

You may replace `{algo}` with `ppo` or `sac` in order to run the scripts.

### Usages

#### Perform hyperparameter optimization

Execute the following command for optimizing hyperparameters

```bash
python ./examples/hpo/hpo_experiment.py \
                    --algo, {algo}, \
                    --task, {sys}, \
                    --overrides, \
                    ./examples/hpo/rl/config_overrides/{sys}/{sys}_{task}.yaml, \
                    ./examples/hpo/rl/{algo}/config_overrides/{sys}/{algo}_{sys}.yaml, \
                    ./examples/hpo/rl/{algo}/config_overrides/{sys}/{algo}_{sys}_hpo.yaml, \
                    --output_dir, {output_dir}, \
                    --seed, 7, \
                    --use_gpu, True
```

You may replace `{sys}` with `cartpole` in order to run the script.

#### Train policy using optimized hyperparameters

Execute the following command if optimized hyperparameters are given

```bash
python ./examples/hpo/hpo_experiment.py \
                    --algo {algo} \
                    --overrides \
                    ./examples/hpo/rl/{algo}/config_overrides/cartpole/{algo}_{sys}.yaml \
                    ./examples/hpo/rl/config_overrides/{sys}/{sys}_{task}.yaml \
                    --output_dir {output_dir} \
                    --tag {run_name} \
                    --opt_hps {best_hp_file} \
                    --task {sys} --seed 2 --use_gpu True
```

You may replace `{sys}` with `cartpole` in order to run the script. As an example, `{best_hp_file}` can be replaced with `examples/hpo/rl/sac/config_overrides/cartpole/optimized_hyperparameters.yaml` for `sac`.
