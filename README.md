# safe-control-gym: IROS 2022 Safe Robot Learning Competition

- Official Webpage: https://www.dynsyslab.org/iros-2022-safe-robot-learning-competition/
- IROS Competition Page: https://iros2022.org/program/competition/#toggle-id-8

## Install on Ubuntu/macOS

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
```

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

### Note:
You may need to separately install `gmp`, a dependency of `pycddlib`:
 ```bash
conda install -c anaconda gmp
 ```
 or 
  ```bash
 sudo apt-get install libgmp-dev
 ```

## Getting Started
Familiarize with APIs and environments with the scripts in [`examples/`](https://github.com/utiasDSL/safe-control-gym/tree/main/examples)
```
$ cd ./examples/                                                                    # Navigate to the examples folder
$ python3 getting_started.py --overrides ./getting_started.yaml
```

See also
```
$ python3 tracking.py --overrides ./tracking.yaml                                   # PID trajectory tracking with the 2D quadcopter
$ python3 verbose_api.py --task cartpole --overrides verbose_api.yaml               #  Printout of the extended safe-control-gym APIs
```


-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute for Artificial Intelligence](https://github.com/VectorInstitute)
