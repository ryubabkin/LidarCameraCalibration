import json

import cv2
import open3d as o3d
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tools.lidar_tools as LT
from tools.camera_tools import find_chessboard_corners_camera, calculate_RT
from tools.plotting import plot_lidar_chessboard


def Calibration_Flow(
        camera_folder: str,
        lidar_folder: str,
        association: pd.DataFrame,
        folder_out: str,
        max_distance: float = 7,
        median_distance_stop: float = 0.004,
        min_delta: float = 0.1,
        cluster_threshold: float = 0.1,
        min_points_in_cluster: int = 40,
        plane_confidence_threshold: float = 0.75,
        plane_inlier_threshold: float = 0.02,
        cb_cells: tuple = (9, 7),
        n_points_interpolate: int = 25000,
        period_resolution: int = 400,
        grid_steps: int = 20,
        grid_threshold: float = 0.6,
):
    with open(f'{camera_folder}/calib.json', 'r') as f:
        camera_params = json.load(f)
    intrinsic, distortion = np.array(camera_params['intrinsic']), np.array(camera_params['distortion'])

    print("=== Lidar background detection started ===")
    background_pcd, stop_id = LT.background_detection(
        association=association,
        folder_in=lidar_folder,
        folder_out=folder_out,
        max_distance=max_distance,
        median_distance_stop=median_distance_stop,
        plot=False
    )
    print("=== Chessboard detection started ===")
    for indx, row in association.iloc[stop_id:].iterrows():
        print(f"Frame {indx} processing...")

        pcd_cloud = o3d.t.io.read_point_cloud(f'{lidar_folder}/{str(indx).zfill(6)}.pcd')
        chessboard, lidar_markers, grid_RT = LT.chessboard_detection(
            background_pcd=background_pcd,
            pcd_cloud=pcd_cloud,
            max_distance=max_distance,
            min_delta=min_delta,
            cluster_threshold=cluster_threshold,
            min_points_in_cluster=min_points_in_cluster,
            plane_confidence_threshold=plane_confidence_threshold,
            plane_inlier_threshold=plane_inlier_threshold,
            cb_cells=cb_cells,
            n_points_interpolate=n_points_interpolate,
            period_resolution=period_resolution,
            grid_steps=grid_steps,
            grid_threshold=grid_threshold,
        )
        if lidar_markers.shape[0] != 0:
            print("    The chessboard was found.")
            print("    Saving data...")
            img = cv2.imread(f'{camera_folder}/undistorted/{str(indx).zfill(6)}.jpg')
            camera_markers = find_chessboard_corners_camera(
                image=img,
                folder_out=folder_out,
                idx=indx
            )
            RT_matrix = calculate_RT(
                lidar_markers=lidar_markers,
                image_markers=camera_markers,
                intrinsic=intrinsic,
                distortion=None,
            )

            LM = np.ones([lidar_markers.shape[0], 4])
            LM[:, :3] = lidar_markers
            image_lidar_markers = (intrinsic @ RT_matrix@(LM.T))
            image_lidar_markers[:2] /= image_lidar_markers[2, :]
            image_lidar_markers = image_lidar_markers.T

            plt.figure(figsize=(7,7))
            plt.scatter(camera_markers[:, 0], camera_markers[:, 1], marker="$\u25EF$", edgecolor='lime', s=125,
                        label='camera')
            plt.scatter(image_lidar_markers[:, 0], image_lidar_markers[:, 1],  marker="$\u25EF$", edgecolor='r', s=125,
                        label='lidar')
            plt.imshow(img)
            plt.legend()
            plt.xlim(camera_markers[:, 0].min() - 50, camera_markers[:, 0].max() + 50)
            plt.ylim(camera_markers[:, 1].max() + 50, camera_markers[:, 1].min() - 50)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            # plot_lidar_chessboard(
            #     chessboard=chessboard,
            #     markers=lidar_markers,
            #     indx=indx,
            #     folder_out=folder_out,
            # )
            print()
            # np.save(f'{folder_out}/data/{str(indx).zfill(6)}.npy', chessboards)


# def transform():
#     LT = LT + np.array([x, y, z])
#     # update rotation matrix
#     LR = utils.Rx(a) @ utils.Ry(b) @ utils.Rz(c) @ LR
#
#     # apply extrinsic to point cloud
#     transformed_points = LR @ points + LT.reshape(-1, 1)
#     depths = transformed_points[2, :]
#     # apply intrinsic to point cloud
#     transformed_points = utils.view_points(transformed_points[:3, :], intrinsic, normalize=True)
