import json
import os

import cv2
import open3d as o3d
import numpy as np
import pandas as pd
import tools.lidar_tools as LT
from tools.association import save_triples, associate_frames, collect_time_stamps
from tools.camera_tools import find_chessboard_corners_camera, calculate_RT, prepare_params_json
from tools.extraction import extract_video_frames, create_output_extraction_folders, extract_rosbag_frames
from tools.plotting import visualize_result


def Calibration_Flow(
        folder: str,
        camera_folder: str,
        association: pd.DataFrame,
        background_pcd: o3d.geometry.PointCloud,
        stop_id: int,
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
        reprojection_error: float = 20
):
    with open(f'{camera_folder}/calib.json', 'r') as f:
        camera_params = json.load(f)
    intrinsic, distortion = np.array(camera_params['intrinsic']), np.array(camera_params['distortion'])
    print("Chessboard detection...")
    lidar_markers_set, camera_markers_set = [], []
    for indx, row in association.iloc[stop_id:].iterrows():
        print(f"    Frame {indx} processing...")
        try:
            pcd_cloud = o3d.t.io.read_point_cloud(f'{folder}/lidar/{str(indx).zfill(6)}.pcd')
            lidar_markers = LT.chessboard_detection(
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
                print("        The chessboard was found.")
                print("        Saving data...")
                img = cv2.imread(f'{camera_folder}/original/{str(indx).zfill(6)}.jpg')
                camera_markers = find_chessboard_corners_camera(
                    image=img
                )
                lidar_markers_set.append(lidar_markers)
                camera_markers_set.append(camera_markers)
        except cv2.error as e:
            print("[ERROR] ", e)
            continue

    # np.save(f'{folder}/lidar_markers.npy', np.array(lidar_markers_set))
    # np.save(f'{folder}/camera_markers.npy', np.array(camera_markers_set))
    lidar_markers = np.vstack(lidar_markers_set)
    camera_markers = np.vstack(camera_markers_set)

    RT_matrix, fraction, rmse, mae = calculate_RT(
        lidar_markers=lidar_markers,
        image_markers=camera_markers,
        intrinsic=intrinsic,
        distortion=distortion,
        reprojection_error=reprojection_error
    )
    print("RT matrix detection fraction: ", fraction)
    print("RT matrix errors (in pixels): RMSE = ", rmse, ", MAE = ", mae)
    print("[DONE] Calibration finished")
    print("Visualization...")
    split = np.array_split(association, 20)
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

    print("[DONE] Visualization finished")
    prepare_params_json(
        folder=camera_folder,
        camera_params=camera_params,
        RT_matrix=RT_matrix,
        elevation=elevation,
        angles=angles
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
        print('Extracting normal camera frames...')
        extract_video_frames(
            filename_in=f'{folder_in}/normal_camera/color.mjpeg',
            output_folder=f'{folder_in}/normal_camera/images/',
            prefix='',
            start_time_sec=0,
            end_time_sec=None
        )
        print('Extracting wide camera frames...')
        extract_video_frames(
            filename_in=f'{folder_in}/wide_camera/color.mjpeg',
            output_folder=f'{folder_in}/wide_camera/images/',
            prefix='',
            start_time_sec=0,
            end_time_sec=None
        )
        print('Extracting lidar frames...')
        rosbag_folder = [folder for folder in os.listdir(f'{folder_in}/lidar/') if folder.startswith("rosbag2")][0]
        extract_rosbag_frames(
            rosbag_folder=f'{folder_in}/lidar/{rosbag_folder}',
            output_folder=f'{folder_in}/lidar/pcd/',
            timestamps_file=f'{folder_in}/lidar/lidar_timestamps.csv',
            prefix=''
        )
        print('[DONE] Extraction is finished')
    lidar, normal, wide = collect_time_stamps(
        folder_in=folder_in,
    )
    print('Association...')
    associated_frames = associate_frames(
        lidar=lidar,
        normal=normal,
        wide=wide,
        n_lag=n_lag_seconds,
        w_lag=w_lag_seconds
    )
    print('Saving...')
    save_triples(
        association=associated_frames,
        folder_in=folder_in,
        folder_out=folder_out
    )
    print('[DONE] Association is finished')
    return associated_frames
