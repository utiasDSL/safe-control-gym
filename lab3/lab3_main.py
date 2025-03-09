import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R


# Camera's intrinsic parameters
K = np.array([
    [698.86, 000.00, 306.91],
    [000.00, 699.13, 150.34],
    [000.00, 000.00, 001.00]
])

# Camera's distortion coefficients
d = np.array([0.191887, -0.563680, -0.0036176, -0.002037, 0.000])

# extrinsic transformation matrix from vehicle body frame to camera frame
T_cb = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# dataset paths
dataset_path = "dataset/"
image_path = dataset_path + "image_folder/output_folder/"
state_dataset_path = dataset_path + "lab3_pose.csv"

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to rotation matrix
    
    :param quaternion: quaternion in the form of [x, y, z, w]
    :return: rotation matrix
    """
    r = R.from_quat(quaternion)
    return r.as_matrix()

def get_height_from_camera(data):
    """
    Get height of the drone from the camera
        
    :param data: dataset containing the pose of the drone
    :return: height of the drone from the ground
    """
    
    quaternion = data[["q_x", "q_y", "q_z", "q_w"]].values
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    translations = data[["p_x", "p_y", "p_z"]].values
    
    # compute rotation matrix
    R_wb = quaternion_to_rotation_matrix(quaternion)
    
    # Transformation from world frame to body frame
    T_wb = np.eye(4)
    T_wb[:3, :3] = R_wb
    T_wb[:3, 3] = translations

    # Calculate tilt angle of the drone relative to Z axis
    tilt_angle = np.arccos(np.dot(R_wb[:, 2], np.array([0, 0, 1])))
    
    # Calculate depth
    height = data[['p_z']].values[0]/np.cos(tilt_angle)

    return height


def main():
    
    state_dataset = pd.read_csv(state_dataset_path)
    
    # append image path to state dataset
    state_dataset["image_name"] = state_dataset.index.to_series().apply(lambda x: f"image_{x}.jpg")
    state_dataset["image_path"] = state_dataset["image_name"].apply(lambda x: image_path + x)
    
    # limit dataset size to 10
    state_dataset = state_dataset.head(900)
        
    
    for i in range(len(state_dataset)):
        # read image
        image = cv2.imread(state_dataset["image_path"][i])
        
        # get height of the drone from the ground
        height = get_height_from_camera(state_dataset.iloc[i])
    
    return 0


if __name__ == "__main__":
    main()
