import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN

from tools.plotting import plot_background_points, plot_training_curve, plot_lidar_chessboard
from tools.utils import choose_best_plane


def background_detection(
        association: pd.DataFrame,
        folder_in: str,
        folder_out: str,
        max_distance: float = 7,
        median_distance_stop: float = 0.004,
        plot: bool = False
):
    print("=== Lidar background detection started ===")
    background_pcd = o3d.geometry.PointCloud()
    points_list, background_points, indx = None, None, 0
    medians, means = [], []
    for indx, row in association.iterrows():
        print(f"Frame {indx} processing...")
        pcd_cloud = o3d.t.io.read_point_cloud(f'{folder_in}/{str(indx).zfill(6)}.pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_cloud.point.positions.numpy())
        intensity = np.asarray(pcd_cloud.point.intensity.numpy())
        points = np.asarray(pcd.points)
        dist = np.sqrt(np.sum(points ** 2, axis=1))
        points, intensity = points[dist < max_distance], intensity[dist < max_distance]

        if plot:
            plot_background_points(
                points=points,
                background_points=np.asarray(background_pcd.points),
                intensity=intensity,
                indx=indx,
                folder_out=folder_out
            )

        pcd.points = o3d.utility.Vector3dVector(points)
        if points_list is None:
            points_list = points
            median = 100
        else:
            distances = np.asarray(pcd.compute_point_cloud_distance(background_pcd))
            median = np.median(distances)
            medians.append(np.median(distances))
            means.append(np.mean(distances))
            points_list = np.concatenate([points_list, points], axis=0)
        background_pcd.points = o3d.utility.Vector3dVector(points_list)
        background_pcd = background_pcd.remove_duplicated_points()
        background_points = np.asarray(background_pcd.points)
        if median < median_distance_stop:
            print(f'Background frames: {indx}, median = {np.round(median, 5)}')
            break

    plot_training_curve(
        medians=np.asarray(medians),
        means=np.asarray(means),
        folder_out=folder_out
    )
    np.save(f'{folder_out}/background_points.npy', background_points)
    return background_pcd, indx + 1


def get_difference(
        pcd_cloud: o3d.geometry.PointCloud,
        background_pcd: o3d.geometry.PointCloud,
        max_distance: float = 7,
        min_delta: float = 0.1,
) -> (np.ndarray, np.ndarray):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_cloud.point.positions.numpy())
    intensity = np.asarray(pcd_cloud.point.intensity.numpy())
    points = np.asarray(pcd.points)
    dist = np.sqrt(np.sum(points ** 2, axis=1))
    points, intensity = points[dist < max_distance], intensity[dist < max_distance]
    pcd.points = o3d.utility.Vector3dVector(points)
    distances = np.asarray(pcd.compute_point_cloud_distance(background_pcd))
    difference, intensity = points[distances > min_delta], intensity[distances > min_delta]
    return difference, intensity


def get_all_obstacles(
        difference: np.ndarray,
        intensity: np.ndarray,
        cluster_threshold: float = 0.1,
        min_points_in_cluster: int = 40,
) -> (list, list):
    obstacle_clouds, obstacle_intensities = [], []

    if len(difference) != 0:
        clusters = DBSCAN(
            eps=cluster_threshold,
            min_samples=min_points_in_cluster
        ).fit(difference)
        obstacle_clouds = [difference[clusters.labels_ == c] for c in np.unique(clusters.labels_) if c != -1]
        obstacle_intensities = [intensity[clusters.labels_ == c] for c in np.unique(clusters.labels_) if c != -1]
    return obstacle_clouds, obstacle_intensities


def select_chessboard(
    obstacle_clouds: list,
    obstacle_intensities: list,
    plane_confidence_threshold: float = 0.75,
    plane_inlier_threshold: float = 0.02,
    median_intensity_threshold: float = None
) -> list:
    chessboards = []
    for idx, obstacle in enumerate(obstacle_clouds):
        obstacle_intensity = obstacle_intensities[idx]
        obstacle_mask = IQR_mask(cloud=obstacle)
        obstacle_intensity = obstacle_intensity[obstacle_mask]
        obstacle = obstacle[obstacle_mask]

        mask, equation, confidence = choose_best_plane(
            points=obstacle,
            inlier_threshold=plane_inlier_threshold
        )
        obstacle = obstacle[mask]
        obstacle_intensity = obstacle_intensity[mask].flatten()
        if confidence > plane_confidence_threshold:
            if median_intensity_threshold is None:
                median_intensity_threshold = np.median(obstacle_intensity)
            print(
                f'confidence: {np.round(confidence, 4)}, '
                f'median_intensity_threshold: {np.round(median_intensity_threshold, 4)}'
            )
            chessboards.append({
                'cloud' : obstacle,
                'intensity' : obstacle_intensity,
                'confidence' : confidence,
                'median_intensity_threshold' : median_intensity_threshold,
                'equation' : equation
            })
    return chessboards


def chessboard_detection(
        background_pcd: o3d.geometry.PointCloud,
        association: pd.DataFrame,
        background_index: int,
        folder_in: str,
        folder_out: str,
        max_distance: float = 7,
        min_delta: float = 0.1,
        cluster_threshold: float = 0.1,
        min_points_in_cluster: int = 40,
        plane_confidence_threshold: float = 0.75,
        plane_inlier_threshold: float = 0.02,
        plot: bool = False
):
    print("=== Lidar chessboard detection started ===")
    background_points = np.asarray(background_pcd.points)
    for indx, row in association.iloc[background_index:].iterrows():
        print(f"Frame {indx} processing...")
        #
        pcd_cloud = o3d.t.io.read_point_cloud(f'{folder_in}/{str(indx).zfill(6)}.pcd')
        difference, intensity = get_difference(
            pcd_cloud=pcd_cloud,
            background_pcd=background_pcd,
            max_distance=max_distance,
            min_delta=min_delta
        )
        obstacle_clouds, obstacle_intensities = get_all_obstacles(
            difference=difference,
            intensity=intensity,
            cluster_threshold=cluster_threshold,
            min_points_in_cluster=min_points_in_cluster
        )
        chessboards = select_chessboard(
            obstacle_clouds=obstacle_clouds,
            obstacle_intensities=obstacle_intensities,
            plane_confidence_threshold=plane_confidence_threshold,
            plane_inlier_threshold=plane_inlier_threshold
        )
        if plot:
            plot_lidar_chessboard(
                chessboards=chessboards,
                background_points=background_points,
                indx=indx,
                folder_out=folder_out
            )


def IQR_mask(
    cloud: np.ndarray,
):
    Q75 = np.quantile(cloud, 0.75, axis=0)
    Q25 = np.quantile(cloud, 0.25, axis=0)
    IQR = Q75 - Q25
    mask = np.ones(cloud.shape[0], dtype=bool)
    mask = np.logical_and(mask, cloud[:, 0] < Q75[0] + 1.5 * IQR[0])
    mask = np.logical_and(mask, (cloud[:, 0] > Q25[0] - 1.5 * IQR[0]))
    mask = np.logical_and(mask, (cloud[:, 1] < Q75[1] + 1.5 * IQR[1]))
    mask = np.logical_and(mask, (cloud[:, 1] > Q25[1] - 1.5 * IQR[1]))
    mask = np.logical_and(mask, (cloud[:, 2] < Q75[2] + 1.5 * IQR[2]))
    mask = np.logical_and(mask, (cloud[:, 2] > Q25[2] - 1.5 * IQR[2]))
    return mask
