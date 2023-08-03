

import os
import pandas as pd


from tools.flows import Calibration_Flow

# %%

folder_in = '/Users/brom/Laboratory/GlobalLogic/MEAA/LidarCameraCalibration/data/third/RESULT'
folder_out = f'{folder_in}/output'
if not os.path.exists(folder_out):
    os.mkdir(folder_out)
if not os.path.exists(f'{folder_out}/lidar'):
    os.mkdir(f'{folder_out}/lidar')
if not os.path.exists(f'{folder_out}/images'):
    os.mkdir(f'{folder_out}/images')
if not os.path.exists(f'{folder_out}/data'):
    os.mkdir(f'{folder_out}/data')

normal_folder = f'{folder_in}/normal_camera/'
wide_folder = f'{folder_in}/wide_camera/'
lidar_folder = f'{folder_in}/lidar/'
association = pd.read_csv(f'{folder_in}/association.csv')

# %%
min_delta = 0.1
max_distance = 7
median_distance_stop = 0.004
cluster_threshold = 0.1
min_points_in_cluster = 100
plane_confidence_threshold = 0.75
plane_inlier_threshold = 0.02


Calibration_Flow(
    camera_folder=normal_folder,
    lidar_folder=lidar_folder,
    association=association,
    folder_out=folder_out,
    max_distance=max_distance,
    median_distance_stop=median_distance_stop,
    min_delta=min_delta,
    cluster_threshold=cluster_threshold,
    min_points_in_cluster=min_points_in_cluster,
)
