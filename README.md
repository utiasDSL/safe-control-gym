# IROS 2022 Safe Robot Learning Competition

Links:
- [Official Webpage](https://www.dynsyslab.org/iros-2022-safe-robot-learning-competition/)
- [IROS Competition Page](https://iros2022.org/program/competition/#toggle-id-8)
- [GitHub Discussions](https://github.com/utiasDSL/safe-control-gym/discussions/categories/iros-2022-competition)

## Registration Form

*Note: alpha release subject to change throughout the month of August, 2022; register for updates*

Complete this [Google Form](https://forms.gle/vEmVK99n1SyaE4Zw9) to register you interest and receive e-mail updates

## Latest Update: August 15
- Collisions count

## Next Update: August 18 (approx.)
- Goal position reached, gate progress metrics
- Training script template
- `cffirmare` Python module

## Install on Ubuntu/macOS

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
```

Create and access a Python 3.8 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n safe python=3.8
conda activate safe
```

Install the `safe-control-gym` repository 

```
pip install --upgrade pip
pip install -e .
```

## Getting Started
Run the scripts in [`competition/`](https://github.com/utiasDSL/safe-control-gym/tree/main/competition)
```
$ cd ./competition/
$ python3 getting_started.py --overrides ./getting_started.yaml
```
**Modify file [`edit_this.py`](https://github.com/utiasDSL/safe-control-gym/blob/alpha-iros-competition/competition/edit_this.py) to customize your planning and control**

-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute for Artificial Intelligence](https://github.com/VectorInstitute)
