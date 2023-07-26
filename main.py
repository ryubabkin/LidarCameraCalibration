
from tools.lidar_tools import background_detection, chessboard_detection

import os
import pandas as pd

# %%

folder_in = '/Users/brom/Laboratory/GlobalLogic/MEAA/LidarCameraCalibration/data/third/RESULT'
folder_out = f'{folder_in}/output'
if not os.path.exists(folder_out):
    os.mkdir(folder_out)
normal_folder = f'{folder_in}/normal_camera/'
wide_folder = f'{folder_in}/wide_camera/'
lidar_folder = f'{folder_in}/lidar/'
association = pd.read_csv(f'{folder_in}/association.csv')


min_delta = 0.1
max_distance = 7
median_distance_stop = 0.004
cluster_threshold = 0.1
min_points_in_cluster = 100
plane_confidence_threshold = 0.75
plane_inlier_threshold = 0.02
# %%

background_pcd, stop_id = background_detection(
    association=association,
    folder_in=lidar_folder,
    folder_out=folder_out,
    max_distance=max_distance,
    median_distance_stop=median_distance_stop,
    plot=False
)

chessboard_detection(
    background_pcd=background_pcd,
    association=association,
    background_index=stop_id,
    folder_in=lidar_folder,
    folder_out=folder_out,
    max_distance=max_distance,
    min_delta=min_delta,
    cluster_threshold=cluster_threshold,
    min_points_in_cluster=min_points_in_cluster,
    plane_confidence_threshold=plane_confidence_threshold,
    plane_inlier_threshold=plane_inlier_threshold,
    plot=True
)
