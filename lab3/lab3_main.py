import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

def convert_pixel_to_world(pixel, height, T_wb):
    """
    Convert pixel coordinates to world coordinates
    
    :param pixel: pixel coordinates
    :param height: height of the drone from the ground
    :param T_wb: transformation matrix from world frame to body frame
    
    :return: world coordinates
    """
        
    # convert pixel to camera frame
    pixel_camera = np.array([pixel[0], pixel[1], 1])
    camera_coordinates = np.dot(np.linalg.inv(K), pixel_camera)
    camera_coordinates = camera_coordinates * height
    
    # transform camera coordinates to body frame
    body_coordinates = np.dot(T_cb, np.append(camera_coordinates, 1))
     
    world_coordinates = np.dot(T_wb, body_coordinates)
     
    return world_coordinates[:2]
    
def get_height_from_camera(height, R_wb):
    """
    Get height of the drone from the camera
        
    :param height: height of the drone from the ground
    :param R_wb: rotation matrix from world frame to body frame
    
    :return: height of the drone from the ground
    """

    # Calculate tilt angle of the drone relative to Z axis
    tilt_angle = np.arccos(np.dot(R_wb[:, 2], np.array([0, 0, 1])))
    
    # Calculate depth
    height = height/np.cos(tilt_angle)

    return height

def detect_target_marker(image):
    """
    Detect target marker in the image
    
    :param image: image containing the target marker
    :return: detected target coordinates in pixel
    """
    targets = []
    
    undistorted_image = cv2.undistort(image, K, d)
    hsv_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)
    
    # define range of green color in HSV
    lower_green = np.array([35, 70, 20])
    upper_green = np.array([80, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # remove noise
    blur = cv2.medianBlur(mask, 5)
    
    # find contours
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Compute perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Compute area
        area = cv2.contourArea(contour)
        
        # Avoid division by zero
        if perimeter == 0:
            continue
        
        # Compute circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Check if the contour is approximately circular
        if circularity > 0.7:  # Threshold close to 1 (ideal circle)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # check if radius is within the range
            if 10 < radius <= 100:
                # draw circle around the target marker
                # cv2.circle(image, center, radius, (0, 255, 0), 2)
                targets.append(center)
        
    return targets

def get_targets(targets):
    """
    Get 6 unique targets from the detected targets using k-means clustering
    
    :param targets: detected targets
    
    :return: 6 unique targets
    :return: cluster labels
    """
    
    if len(targets) < 6:
        return targets
    
    kmeans = KMeans(n_clusters=6, random_state=0).fit(targets)
    centers = kmeans.cluster_centers_
    
    return centers, kmeans.labels_
    

def main():
    
    state_dataset = pd.read_csv(state_dataset_path)
    
    # append image path to state dataset
    state_dataset["image_name"] = state_dataset.index.to_series().apply(lambda x: f"image_{x}.jpg")
    state_dataset["image_path"] = state_dataset["image_name"].apply(lambda x: image_path + x)
    
    # limit dataset size to 10
    #state_dataset = state_dataset.head(900)
    targets = np.array([])
        
    
    for i in range(len(state_dataset)):
        data = state_dataset.iloc[i]
        
        # read image
        image = cv2.imread(data["image_path"])
    
        quaternion = data[["q_x", "q_y", "q_z", "q_w"]].values
        translations = data[["p_x", "p_y", "p_z"]].values
        
        # compute rotation matrix
        R_wb = quaternion_to_rotation_matrix(quaternion)
        
        # Transformation from world frame to body frame
        T_wb = np.eye(4)
        T_wb[:3, :3] = R_wb
        T_wb[:3, 3] = translations
    
        
        # get height of the drone from the ground
        height = get_height_from_camera(data["p_z"], R_wb)
        
        # detect target marker in the image
        target = detect_target_marker(image)
        
        for t in target:
            world_coordinates =  convert_pixel_to_world(t, height, T_wb)
            targets = np.append(targets, world_coordinates)
        # save image for debugging, output/images/
        # cv2.imwrite(f"output/images/image_{i}.jpg", image)
    
    targets = targets.reshape(-1, 2)
    main_targets, cluster_labels = get_targets(targets)
    
    # Create figure
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", len(main_targets))
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=targets[:, 0], y=targets[:, 1], hue=cluster_labels, palette=palette, alpha=0.3, edgecolor=None)
    
    sns.scatterplot(x=main_targets[:, 0], y=main_targets[:, 1], color="black", s=120, marker="X", label="Cluster Centers")
    
    for i, (x, y) in enumerate(main_targets):
        plt.text(x+0.25, y + 0.15, f"({x:.2f}, {y:.2f})", fontsize=10, ha='right', va='bottom', color=palette[i])

    plt.xlim(-1.5, 2.0)
    plt.ylim(-2.5, 1.5)
    # Add labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Detected Targets")
    # remove legend
    plt.legend().remove()
    plt.savefig("detected_targets.png", dpi=300)
    
    return 0


if __name__ == "__main__":
    main()
