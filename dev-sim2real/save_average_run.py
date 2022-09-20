import argparse

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from trial_data_utils import get_data, get_average_run


parser = argparse.ArgumentParser()
parser.add_argument("run")
args = parser.parse_args()

trials, header_map = get_data(args.run)

avg_trial = get_average_run(trials)
np.savetxt(f"{args.run}/data/average_run.csv", avg_trial, delimiter=",", header="time,x,y,z,qx,qy,qz,qw")

fig = plt.figure()
ax = plt.axes(projection='3d')

xline = avg_trial[:,1]
yline = avg_trial[:,2]
zline = avg_trial[:,3]
ax.plot3D(xline, yline, zline)

plt.show()
