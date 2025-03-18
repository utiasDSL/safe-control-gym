# README: Target Detection from Drone Images

## Overview
This script processes images captured by a drone and detects target markers using image processing techniques. It then converts the detected pixel coordinates to world coordinates using camera intrinsics and extrinsics, and clusters the targets using K-Means. Finally, it visualizes the detected targets.

## Requirements
Ensure you have the following dependencies installed before running the script:

```sh
pip install numpy pandas opencv-python scipy scikit-learn matplotlib seaborn
```

## Usage

1. Place the dataset (pose data and images) inside the `dataset/` directory.
2. Ensure `lab3_pose.csv` (state dataset) is in the `dataset/` folder.
3. Run the script:

```sh
python script.py
```

4. The script will process the images and generate `detected_targets.png`, displaying the detected targets in world coordinates.

## Input Data Format
- **Pose Data (`lab3_pose.csv`)**: The CSV file should include the following columns:
  - `p_x, p_y, p_z`: Position of the drone in world coordinates.
  - `q_x, q_y, q_z, q_w`: Quaternion representing the drone's orientation.
- **Images**: Images should be named sequentially as `image_0.jpg`, `image_1.jpg`, etc.

## Output
- `detected_targets.png`: A scatter plot showing detected targets with cluster centers.

## List of Functions

### 1. `quaternion_to_rotation_matrix(quaternion)`
Converts a quaternion `[x, y, z, w]` into a rotation matrix.

### 2. `convert_pixel_to_world(pixel, height, T_wb)`
Converts pixel coordinates from the image to world coordinates.

### 3. `get_height_from_camera(height, R_wb)`
Adjusts the drone height based on its tilt angle.

### 4. `detect_target_marker(image)`
Detects circular green markers in the image and returns their pixel coordinates.

### 5. `get_targets(targets)`
Clusters detected targets using K-Means and returns 6 unique targets.

### 6. `main()`
Main execution function that reads the dataset, detects targets, converts coordinates, and generates the output visualization.

## Notes
- Images should be stored in `dataset/image_folder/output_folder/`, named sequentially as `image_0.jpg`, `image_1.jpg`, etc.
- The CSV file `lab3_pose.csv` should be placed inside the `dataset/` directory.
- Ensure your dataset is structured correctly before running the script.
- Adjust color thresholds in `detect_target_marker()` if needed for different lighting conditions.
- The script assumes a fixed camera intrinsic matrix and transformation matrix (`T_cb`). Modify these values if your setup differs.
- Ensure your dataset is structured correctly before running the script.
- Adjust color thresholds in `detect_target_marker()` if needed for different lighting conditions.
- The script assumes a fixed camera intrinsic matrix and transformation matrix (`T_cb`). Modify these values if your setup differs.

