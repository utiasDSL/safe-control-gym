import argparse

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from trial_data_utils import get_data

parser = argparse.ArgumentParser()
parser.add_argument("run")
args = parser.parse_args()

trials, header_map = get_data(args.run)

fig = plt.figure()
ax = plt.axes(projection='3d')

for trial in trials:
    vicon_idxs = list(set(np.where(trial[:,header_map["vicon_pos_x"]:header_map["vicon_orientation_w"]+1] != 0)[0]))
    xline = trial[vicon_idxs,header_map["vicon_pos_x"]]
    yline = trial[vicon_idxs,header_map["vicon_pos_y"]]
    zline = trial[vicon_idxs,header_map["vicon_pos_z"]]
    ax.plot3D(xline, yline, zline)

plt.show()
