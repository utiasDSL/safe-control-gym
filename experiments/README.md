# Sim2Real Experiments

To run simulated trajectories:

```bash
cd experiments/
python3 getting_started.py --run=<run>
```

Where \<run\> can be any experiment subfolder name containing `edit_this.py` controller and `getting_started.yaml` configuration file.
Included examples are:

- `ellipse`
- `hypotrochoid`
- `line`
- `lissajous`
- `outward_spiral`
- `outward_spiral_varying_z`
- `slalom`
- `torus`
- `torus_bodyRates`
- `torus_cmdFullState`
- `zig_zag_climb`
- `zig_zag_fall`

A number of additional tools are included in this branch for data analysis of real world data. 

* `bag_to_csv.py`—converts rosbag files to a more readable csv format
* `save_average_run.py`—given csv files collected from real world, produce an average run for comparison to sim performance. We use a sliding time window to gather an average position point from multiple trajectories. 
* `sim_data_utils.py`—contains tools for loading real world data into sim 
* `trial_data_utils.py`—contains tools for processing real world data 
* `view_trial.py`—views real world data trial 
