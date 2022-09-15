import numpy as np

def load_average_run(run):
    with open(f"{run}/data/average_run.csv", "r") as f:
        raw_data = f.readlines() 
        header = raw_data[0].strip('\n').split(',')
        data = []
        for line in raw_data[1:]:
            time, x, y, z, qx, qy, qz, qw = [float(item) for item in line.strip('\n').split(',')]
            data += [[time, x, y, z, qx, qy, qz, qw]]
    return np.array(data)
        