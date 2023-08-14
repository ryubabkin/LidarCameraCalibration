import json
import os
import sys

import cv2
import open3d as o3d
import numpy as np
import pandas as pd
import tools.lidar_tools as LT
from tools.association import save_triples, associate_frames, collect_time_stamps
from tools.camera_tools import find_chessboard_corners_camera, calculate_RT, prepare_params_json
from tools.extraction import extract_video_frames, create_output_extraction_folders, extract_rosbag_frames
from tools.plotting import visualize_result
from tools.utils import LogsException

LogExc = LogsException()


def Calibration_Flow(
        folder: str,
        camera_folder: str,
        association: pd.DataFrame,
        background_pcd: o3d.geometry.PointCloud,
        stop_id: int,
        hfov: float,
        vfov: float,
        elevation: float = 0.0,
        angles: tuple = (0.0, 0.0, 0.0),
        max_distance: float = 7,
        min_delta: float = 0.1,
        cluster_threshold: float = 0.1,
        min_points_in_cluster: int = 40,
        min_points_in_chessboard: int = 300,
        plane_confidence_threshold: float = 0.75,
        plane_inlier_threshold: float = 0.02,
        cb_cells: tuple = (9, 7),
        n_points_interpolate: int = 25000,
        period_resolution: int = 400,
        grid_steps: int = 20,
        grid_threshold: float = 0.55,
        reprojection_error: float = 20,
        n_images_to_draw: int = 20
):
    with open(f'{camera_folder}/calib.json', 'r') as f:
        camera_params = json.load(f)
    intrinsic, distortion = np.array(camera_params['intrinsic']), np.array(camera_params['distortion'])
    LogExc.start("Chessboard detection...")
    lidar_markers_set, camera_markers_set, points_per_cb = [], [], []
    for indx, row in association.iloc[stop_id:].iterrows():
        LogExc.info(f"Frame {indx} processing...")
        try:
            pcd_cloud = o3d.t.io.read_point_cloud(f'{folder}/lidar/{str(indx).zfill(6)}.pcd')
            lidar_markers, n_points = LT.chessboard_detection(
                background_pcd=background_pcd,
                pcd_cloud=pcd_cloud,
                max_distance=max_distance,
                min_delta=min_delta,
                cluster_threshold=cluster_threshold,
                min_points_in_cluster=min_points_in_cluster,
                min_points_in_chessboard=min_points_in_chessboard,
                plane_confidence_threshold=plane_confidence_threshold,
                plane_inlier_threshold=plane_inlier_threshold,
                cb_cells=cb_cells,
                n_points_interpolate=n_points_interpolate,
                period_resolution=period_resolution,
                grid_steps=grid_steps,
                grid_threshold=grid_threshold,
            )
            if lidar_markers.shape[0] != 0:
                LogExc.info("The chessboard was detected. Processing...")
                img = cv2.imread(f'{camera_folder}/original/{str(indx).zfill(6)}.jpg')
                camera_markers = find_chessboard_corners_camera(
                    image=img
                )
                lidar_markers_set.append(lidar_markers)
                camera_markers_set.append(camera_markers)
                points_per_cb.append(n_points)
        except cv2.error:
            LogExc.error("OpenCV error")
            continue

    # np.save(f'{folder}/lidar_markers.npy', np.array(lidar_markers_set))
    # np.save(f'{folder}/camera_markers.npy', np.array(camera_markers_set))
    lidar_markers = np.vstack(lidar_markers_set)
    camera_markers = np.vstack(camera_markers_set)
    distance = np.linalg.norm(lidar_markers, axis=1)
    RT_matrix, fraction, error = calculate_RT(
        lidar_markers=lidar_markers,
        image_markers=camera_markers,
        intrinsic=intrinsic,
        distortion=distortion,
        reprojection_error=reprojection_error,
        hfov=hfov,
        vfov=vfov,
        image_size=(1920, 1080)
    )
    avg_distance = np.mean(distance)
    avg_points_per_cb = np.mean(points_per_cb) / (cb_cells[0]-1) / (cb_cells[1]-1)
    LogExc.info(f"Avg. distance to the chessboard: {np.round(avg_distance, 3)} meters")
    LogExc.info(f"Avg. number of lidar points per chessboard cell: {np.round(avg_points_per_cb, 3)}")
    LogExc.info(f"RT matrix detection fraction: {np.round(fraction, 3)}")
    LogExc.info(
        f"RT matrix errors (in pixels): "
        f"RMSE = {np.round(error[0], 5)}, "
        f"MAE = {np.round(error[1],5)}"
    )
    LogExc.info(
        f"RT matrix errors (in degrees): "
        f"RMSE = {np.round(error[2], 5)}, "
        f"MAE = {np.round(error[3],5)}"
    )
    LogExc.info("Visualization...")
    split = np.array_split(association, n_images_to_draw)
    for subset in split:
        id = subset.index[0]
        visualize_result(
            folder=folder,
            camera_folder=camera_folder,
            indx=id,
            intrinsic=intrinsic,
            distortion=distortion,
            RT_matrix=RT_matrix
        )
    prepare_params_json(
        folder=camera_folder,
        camera_params=camera_params,
        RT_matrix=RT_matrix,
        elevation=elevation,
        angles=angles,
        error=error,
        points=avg_points_per_cb,
        distance=float(avg_distance)
    )


def Association_Flow(
        folder_in: str,
        folder_out: str,
        n_lag_seconds: float = 0.0,
        w_lag_seconds: float = 0.0,
        extract: bool = True,
) -> pd.DataFrame:
    if extract:
        create_output_extraction_folders(
            folder_out=folder_in
        )
        LogExc.start('Extracting...')
        if os.path.exists(f'{folder_in}/normal_camera') and os.path.exists(f'{folder_in}/normal_camera/color.mjpeg'):
            LogExc.info('Extracting normal camera frames...')
            extract_video_frames(
                filename_in=f'{folder_in}/normal_camera/color.mjpeg',
                output_folder=f'{folder_in}/normal_camera/images/',
                prefix='',
                start_time_sec=0,
                end_time_sec=None
            )
        else:
            LogExc.warn("Normal camera video file does not exist")
        if os.path.exists(f'{folder_in}/wide_camera') and os.path.exists(f'{folder_in}/wide_camera/color.mjpeg'):
            LogExc.info('Extracting wide camera frames...')
            extract_video_frames(
                filename_in=f'{folder_in}/wide_camera/color.mjpeg',
                output_folder=f'{folder_in}/wide_camera/images/',
                prefix='',
                start_time_sec=0,
                end_time_sec=None
            )
        else:
            LogExc.warn("Wide camera video file does not exist")
        rosbag_folder = [folder for folder in os.listdir(f'{folder_in}/lidar/') if folder.startswith("rosbag2")][0]
        if os.path.exists(f'{folder_in}/lidar') and os.path.exists(rosbag_folder):
            LogExc.info('Extracting lidar frames...')
            extract_rosbag_frames(
                rosbag_folder=f'{folder_in}/lidar/{rosbag_folder}',
                output_folder=f'{folder_in}/lidar/pcd/',
                timestamps_file=f'{folder_in}/lidar/lidar_timestamps.csv',
                prefix=''
            )
        else:
            LogExc.error("Lidar rosbag file does not exist")
            sys.exit(1)
    LogExc.done('Extraction is finished')
    lidar, normal, wide = collect_time_stamps(
        folder_in=folder_in
    )
    LogExc.start('Association...')
    associated_frames = associate_frames(
        lidar=lidar,
        normal=normal,
        wide=wide,
        n_lag=n_lag_seconds,
        w_lag=w_lag_seconds
    )
    LogExc.info('Saving...')
    save_triples(
        association=associated_frames,
        folder_in=folder_in,
        folder_out=folder_out
    )
    LogExc.done('Association is finished')
    return associated_frames
