# Instruction for running hyperparameter optimization

## Install on Ubuntu

### Create a `conda` environment

Create the environment by following the steps presented in the main
[`README.md`](README.md)


You may replace `{algo}` with, e.g., `ppo`, in order to run the scripts.

### Usages

#### Perform hyperparameter optimization

Execute the following command for optimizing hyperparameters

```bash
python ./examples/hpo/hpo_experiment.py \
                    --algo, {algo}, \
                    --task, {sys}, \
                    --overrides, \
                    ./examples/hpo/{sys}/config_overrides/{sys}_{task}.yaml, \
                    ./examples/hpo/{sys}/{algo}/config_overrides/{algo}_{sys}_{task}_{prior}.yaml, \
                    ./examples/hpo/{sys}/{algo}/config_overrides/{algo}_{sys}_hpo.yaml, \
                    --output_dir, {output_dir}, \
                    --seed, 7, \
                    --use_gpu, True
```

You may replace `{sys}` with `cartpole` and `{prior}` with `100`, `200`, or `''`, respectively, in order to run the script.
