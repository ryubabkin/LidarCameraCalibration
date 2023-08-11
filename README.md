LidarCameraCalibration
=======================

## Requirements
1. Python >= 3.8

## Setup process

1. `$ git clone`
2. `$ cd ./{project_folder}`
3. `$ python -m venv env`
4. `$ source env/bin/activate`
5. `$ python -m pip install -r requirements.txt`

## Required input folder structure
```
folder
|_ lidar
|  |_ rosbag2_YYYY_MM_DD-HH_MM_SS
|_ normal_camera
|  |_ color.mjpeg
|  |_ timestamps.json
|  |_ calib.json
|_ wide_camera
   |_ color.mjpeg
   |_ timestamps.json
   |_ calib.json
```

## How to run Calibration / Association Flow:
1. Fill ___settings.json___
```
(C) - for calibration, (A) - for association

MANDATORY

 "chessboard_cells" : [9, 7]     - number of cells in chessboard, [horizontal, vertical] (C)
 "wide_camera_parameters" : {
     "elevation" : 0.0,          - elevation of the camera in meters (C)
     "angles" : [0.0, 0.0, 0.0]  - rotation angles in degrees [pitch, yaw, roll] (C)
 }
 "normal_camera_parameters" : {
     "elevation" : 3.2512,       - elevation of the camera in meters (C)
     "angles" : [10.1, 0.7, 0]   - rotation angles in degrees [pitch, yaw, roll] (C)
 }
 "max_distance" : 10             - max distance for points to be considered for calibration, meters (C)
 "lag_seconds_normal": 0.0,      - lag for normal camera (A, C
 "lag_seconds_wide": 0.0,        - lag for wide camera (A, C)
 
OPTIONAL
 
 "cluster_threshold" : 0.1       - threshold for clustering (C)
 "grid_threshold" : 0.5,         - threshold for grid (C)
 "grid_steps" : 20,              - number of steps for grid (C)
 "median_distance_stop" : 0.004, - background learning curve threshold (C)
 "min_delta" : 0.1,              - distance between points in cluster (C)
 "min_points_in_cluster" : 40,   - min number of points in cluster (C)
 "min_points_in_chessboard" : 300,     - min number of points in chessboard to be considered (C)
 "n_points_interpolate" : 25000,       - number of chessboard points to interpolate (C)
 "period_resolution" : 400,            - chessboard period resolution (C)
 "plane_confidence_threshold" : 0.75,  - threshold for plane confidence (C)
 "plane_inlier_threshold" : 0.02       - threshold for plane inlier points (C)
 "reprojection_error" : 20             - max error distance for RANSAC in RT matrix calibration (C)
```
2. Run ___calibration_tool.py___ from terminal

`$ python calibration_tool.py "{/path/to/settings.json}" "{/path/to/input/folder}" "{/path/to/output/folder}" {do_extraction, 1/0} {do_calibration, 1/0}`

Example: `$ python calibration_tool.py "./settings.json" "./input" "./output" 1 1`

## Output folder structure
```
output
|_ association.csv
|_ background_points.pcd
|_ learning_curve.png
|
|_ lidar
|  |_ 000000.pcd
|  |_ 000001.pcd
|  |_ ....
|
|_ normal_camera
|  |_ original
|  |   |_ 000000.jpg
|  |   |_ 000001.jpg
|  |   |_ ....
|  |_ undistorted
|  |   |_ 000000.jpg
|  |   |_ 000001.jpg
|  |   |_ ....
|  |_ visualization
|  |   |_ .jpg
|  |   |_ .jpg
|  |   |_ ....
|  |_ calib.json
|
|_ wide_camera
   |_ original
   |   |_ 000000.jpg
   |   |_ 000001.jpg
   |   |_ ....
   |_ undistorted
   |   |_ 000000.jpg
   |   |_ 000001.jpg
   |   |_ ....
   |_ visuzlization
   |   |_ .jpg
   |   |_ .jpg
   |   |_ ....
   |_ calib.json
```

## Resulting _calib.json_ structure
```
{
    "lidar_extrinsic": {
        "rotation_x" - x-axis part of rotation quaternion
        "rotation_y" - y-axis part of rotation quaternion
        "rotation_z" - z-axis part of rotation quaternion
        "rotation_w" - real part of rotation quaternion
        "x" - translation along x axis
        "y" - translation along y axis
        "z" - translation along z axis
        "f_x": - focal length x
        "f_y": - focal length y
        "c_x": - principal point x
        "c_y": - principal point y
    },
    "intrinsic"     - camera intrinsic parameters
    "distortion"    - camera distortion parameters
    "elevation"     - camera elevation in meters,
    "angles"        - camera rotation angles in degrees [pitch, yaw, roll]
    "resolution"    - camera resolution [width, height] [1920,1080] 
}
```