import numpy as np
import os, glob


def get_data(run):
    results_folder = glob.glob(f"{run}/data/*/")

    headers = [
        "time",

        "takeoff",
        "land",
        "goto",
        "notifySetpointStop",
        "cmdFullState",

        "cmd_pos_x",
        "cmd_pos_y",
        "cmd_pos_z",
        "cmd_orientation_x",
        "cmd_orientation_y",
        "cmd_orientation_z",
        "cmd_orientation_w",
        "cmd_vel_x",
        "cmd_vel_y",
        "cmd_vel_z",
        "cmd_acc_x",
        "cmd_acc_y",
        "cmd_acc_z",
        "cmd_bodyrates_roll",
        "cmd_bodyrates_pitch",
        "cmd_bodyrates_yaw",

        "vicon_pos_x",
        "vicon_pos_y",
        "vicon_pos_z",
        "vicon_orientation_x",
        "vicon_orientation_y",
        "vicon_orientation_z",
        "vicon_orientation_w",
    ]
    header_map = dict(zip(headers, list(range(len(headers)))))


    trials = []
    for results in results_folder:
        output_data = []
        # ros_out 
        file = "_slash_rosout.csv"

        with open(os.path.join(results, file), 'r') as f:
            raw_data = f.readlines()
            raw_headers = raw_data[0].strip('\n').split(',')

            for line in raw_data[1:]:
                _toadd = [0 for _ in range(len(headers))]

                line = line.strip('\n').split(',')

                _toadd[header_map["time"]] = int(line[4]) + float(line[5])*1e-9 # time 
                msg = line[9]
                if "Takeoff" in msg:
                    _toadd[header_map["takeoff"]] = 1
                elif "Land" in msg:
                    _toadd[header_map["land"]] = 1
                elif "GoTo" in msg:
                    _toadd[header_map["goto"]] = 1
                elif "NotifySetpointsStop" in msg:
                    _toadd[header_map["notifySetpointStop"]] = 1
                else:
                    continue

                output_data += [_toadd]

        # vicon
        file = "_slash_vicon_slash_cf9_slash_cf9.csv"
        flag = False

        with open(os.path.join(results, file), 'r') as f:
            raw_data = f.readlines()
            raw_headers = raw_data[0].strip('\n').split(',')

            for line in raw_data[1:]:
                _toadd = [0 for _ in range(len(headers))]

                line = line.strip('\n').split(',')

                _toadd[header_map["time"]] = int(line[4]) + float(line[5])*1e-9 # time 
                
                x, y, z = line[10:13] # pos
                _toadd[header_map["vicon_pos_x"]] = float(x)
                _toadd[header_map["vicon_pos_y"]] = float(y)
                _toadd[header_map["vicon_pos_z"]] = float(z)
                if not flag:
                    landing_height = float(z)
                    flag = True

                x, y, z, w = line[14:18] # orientation
                _toadd[header_map["vicon_orientation_x"]] = float(x)
                _toadd[header_map["vicon_orientation_y"]] = float(y)
                _toadd[header_map["vicon_orientation_z"]] = float(z)
                _toadd[header_map["vicon_orientation_w"]] = float(w)

                output_data += [_toadd]

        # cmdFullState
        file = "_slash_cf9_slash_cmd_full_state.csv"
        idxs = [4, 5, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 27, 28, 29]

        with open(os.path.join(results, file), 'r') as f:
            raw_data = f.readlines()
            raw_headers = raw_data[0].strip('\n').split(',')

            for line in raw_data[1:]:
                _toadd = [0 for _ in range(len(headers))]

                line = line.strip('\n').split(',')

                _toadd[header_map["time"]] = int(line[4]) + float(line[5])*1e-9 # time 
                
                x, y, z = line[9:12] # pos
                _toadd[header_map["cmd_pos_x"]] = float(x)
                _toadd[header_map["cmd_pos_y"]] = float(y)
                _toadd[header_map["cmd_pos_z"]] = float(z)

                x, y, z = line[19:22] # vel
                _toadd[header_map["cmd_vel_x"]] = float(x)
                _toadd[header_map["cmd_vel_y"]] = float(y)
                _toadd[header_map["cmd_vel_z"]] = float(z)

                x, y, z = line[27:30] # acc
                _toadd[header_map["cmd_acc_x"]] = float(x)
                _toadd[header_map["cmd_acc_y"]] = float(y)
                _toadd[header_map["cmd_acc_z"]] = float(z)

                x, y, z = line[23:26] # body rates
                _toadd[header_map["cmd_bodyrates_roll"]] = float(x)
                _toadd[header_map["cmd_bodyrates_pitch"]] = float(y)
                _toadd[header_map["cmd_bodyrates_yaw"]] = float(z)

                x, y, z, w = line[13:17] # orientation
                _toadd[header_map["cmd_orientation_x"]] = float(x)
                _toadd[header_map["cmd_orientation_y"]] = float(y)
                _toadd[header_map["cmd_orientation_z"]] = float(z)
                _toadd[header_map["cmd_orientation_w"]] = float(w)

                _toadd[header_map["cmdFullState"]] = 1

                output_data += [_toadd]

        output_data_np = np.array(output_data)
        output_data_np = output_data_np[output_data_np[:,0].argsort()]

        # remove all data gathered prior to takeoff command being sent 
        takeoff_idx = np.where(output_data_np[:,header_map["takeoff"]])[0][0]

        output_data_np = output_data_np[takeoff_idx:]
        output_data_np[:,header_map["time"]] -= output_data_np[0,header_map["time"]]


        
        # remove all data gathered after land event is finished 
        flying_idxs = np.where(output_data_np[:,header_map["vicon_pos_z"]] > landing_height*1.05)[0]
        output_data_np = output_data_np[:flying_idxs[-1]]

        trials += [output_data_np]

    return trials, header_map