import random

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN
import shapely as shp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from tools.plotting import plot_training_curve
from tools.utils import choose_best_plane, vec2vec, sort_4gram, Rz


def background_detection(
        association: pd.DataFrame,
        folder: str,
        max_distance: float = 7,
        median_distance_stop: float = 0.004
):
    background_pcd = o3d.geometry.PointCloud()
    points_list, background_points, indx = None, None, 0
    medians, means = [], []
    for indx, row in association.iterrows():
        print(f"    Frame {indx} processing...")
        pcd_cloud = o3d.t.io.read_point_cloud(f'{folder}/lidar/{str(indx).zfill(6)}.pcd')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_cloud.point.positions.numpy())
        intensity = np.asarray(pcd_cloud.point.intensity.numpy())
        points = np.asarray(pcd.points)
        dist = np.sqrt(np.sum(points ** 2, axis=1))
        points, intensity = points[dist < max_distance], intensity[dist < max_distance]
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
        folder_out=folder
    )
    np.save(f'{folder}/background_points.npy', background_points)
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


def detect_chessboards(
        clouds: list,
        intensities: list,
        plane_confidence_threshold: float = 0.8,
        plane_inlier_threshold: float = 0.02,
        min_points_in_chessboard: int = 300,
        median_intensity_threshold: float = None
) -> list:
    chessboards = []
    for idx, cloud in enumerate(clouds):
        cloud_intensity = intensities[idx]
        cloud_mask = IQR_mask(cloud=cloud)
        cloud_intensity = cloud_intensity[cloud_mask]
        cloud = cloud[cloud_mask]

        mask, equation, confidence = choose_best_plane(
            points=cloud,
            inlier_threshold=plane_inlier_threshold
        )
        cloud = cloud[mask]
        cloud_intensity = cloud_intensity[mask].flatten()

        if (confidence > plane_confidence_threshold) & (cloud.shape[0] >= min_points_in_chessboard):
            if median_intensity_threshold is None:
                median_intensity_threshold = np.median(cloud_intensity)
            print(
                f'        Confidence: {np.round(confidence, 4)}, '
                f'Shape: {cloud.shape}, '
                f'Intensity threshold: {np.round(median_intensity_threshold, 4)}'
            )
            intensity_mask = get_chessboard_intensity_mask(cloud_intensity)
            cloud = np.hstack([cloud, intensity_mask.reshape(-1, 1)])
            chessboards.append({
                'cloud': cloud,
                'confidence': confidence,
                'equation': equation
            })
    return chessboards


def chessboard_detection(
        background_pcd: o3d.geometry.PointCloud,
        pcd_cloud: o3d.geometry.PointCloud,
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
        grid_threshold: float = 0.6,
) -> np.ndarray:
    cb_cells = (cb_cells[0]+1, cb_cells[1]+1)
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
    chessboards = detect_chessboards(
        clouds=obstacle_clouds,
        intensities=obstacle_intensities,
        plane_confidence_threshold=plane_confidence_threshold,
        plane_inlier_threshold=plane_inlier_threshold,
        min_points_in_chessboard=min_points_in_chessboard
    )
    if len(chessboards) != 0:
        chessboard = max(chessboards, key=lambda x: x['cloud'].shape[0])
        cloud, equation = chessboard['cloud'], chessboard['equation']
        corrected, RT = lidar_chessboard_RT(
            cloud=cloud[:, :3],
            plane_equation=equation
        )
        cloud[:, :3] = corrected
        cloud, intensity_mask = interpolate_lidar_chessboard(
            cloud=cloud[:, :3],
            intensity_mask=cloud[:, 3] > 0,
            n_points=n_points_interpolate
        )

        rotated, horizontal, vertical, RT = get_cell_period(
            points=cloud,
            intensity_mask=intensity_mask,
            RT=RT,
            resolution=period_resolution
        )
        grid, coeff, diff = get_lidar_grid(
            image=rotated,
            shape=cb_cells,
            vertical=vertical,
            horizontal=horizontal,
            intensity_mask=intensity_mask,
            steps=grid_steps
        )
        print(f"        Correlation: {coeff}, Difference: {diff}")
        if coeff > grid_threshold:
            grid = crop_grid(grid=grid)
            X, Y = np.meshgrid(grid[0], grid[1])
            lidar_markers = np.vstack([X.ravel(), Y.ravel()]).T
            lidar_markers = np.hstack([lidar_markers, np.ones([len(lidar_markers), 1]) * rotated[:, 2].mean()])
            lidar_markers = (np.linalg.inv(RT) @ lidar_markers.T).T
            return lidar_markers
    return np.array([])


def crop_grid(grid: list[np.ndarray, np.ndarray]) -> list[np.ndarray, np.ndarray]:
    grid_x, grid_y = grid
    min_x, max_x = grid_x.min(), grid_x.max()
    min_y, max_y = grid_y.min(), grid_y.max()
    grid_x = grid_x[np.logical_and(grid_x > min_x, grid_x < max_x)]
    grid_y = grid_y[np.logical_and(grid_y > min_y, grid_y < max_y)]
    return [grid_x, grid_y]


def IQR_mask(
        cloud: np.ndarray,
) -> np.ndarray:
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


def lidar_chessboard_RT(
        cloud: np.ndarray,
        plane_equation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    RT = vec2vec(plane_equation[:3], np.array([0, 0, 1]))
    RT = Rz(90) @ RT
    rotated = (RT.dot(cloud.T)).T
    corrected, theta = get_correct_rectangle(points=rotated)
    RT = Rz(theta) @ RT
    return corrected, RT


def get_chessboard_intensity_mask(
        intensity: np.ndarray,
        intensity_threshold: float = None,
) -> np.ndarray:
    if intensity_threshold is None:
        intensity_threshold = np.median(intensity)
    mask = np.ones(intensity.shape[0], dtype=bool)
    mask = np.logical_and(mask, intensity > intensity_threshold)
    return mask


def get_lidar_envelope(
        cloud: np.ndarray,
        intensity_mask: np.ndarray
) -> tuple[shp.Polygon, shp.Polygon]:
    blacks = cloud[~intensity_mask]
    whites = cloud[intensity_mask]

    black_points = shp.MultiPoint(blacks)
    white_points = shp.MultiPoint(whites)

    black_envelope = black_points.oriented_envelope
    white_envelope = white_points.oriented_envelope
    return black_envelope, white_envelope


def interpolate_lidar_chessboard(
        cloud: np.ndarray,
        intensity_mask: np.ndarray,
        n_points: int = 50000
) -> tuple[np.ndarray, np.ndarray]:
    black_envelope, white_envelope = get_lidar_envelope(
        cloud=cloud,
        intensity_mask=intensity_mask,
    )
    envelope = white_envelope if white_envelope.area < black_envelope.area else black_envelope

    points = []
    x_min, x_max = np.min(envelope.exterior.xy[0]), np.max(envelope.exterior.xy[0])
    y_min, y_max = np.min(envelope.exterior.xy[1]), np.max(envelope.exterior.xy[1])
    while len(points) < n_points:
        point = [random.uniform(x_min, x_max), random.uniform(y_min, y_max), cloud[:, 2].mean()]
        pnt = shp.Point(point)
        if envelope.contains(pnt):
            points.append(point)

    coordinates = np.array(points)
    Y = np.where(intensity_mask, 1, 0)
    model_1 = KNeighborsClassifier().fit(cloud, Y)
    interpolated_mask = model_1.predict(coordinates)
    model_2 = HistGradientBoostingClassifier().fit(coordinates, interpolated_mask)
    interpolated_mask = model_2.predict(coordinates)

    return coordinates, interpolated_mask.astype(bool)


def get_correct_rectangle(
        points: np.ndarray,
) -> tuple[np.ndarray, float]:
    vertices = sort_4gram(pts=points[:, :2])
    theta = np.rad2deg(np.arctan((vertices[1, 0] - vertices[0, 0]) / (vertices[1, 1] - vertices[0, 1])))
    rotated = (Rz(theta) @ points.T).T
    return rotated, theta


def get_cell_period(
        points: np.ndarray,
        intensity_mask: np.ndarray,
        RT: np.ndarray,
        resolution: int = 100,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    rotated = points.copy()
    rotated[:, :3], theta = get_correct_rectangle(points=rotated[:, :3])
    RT = Rz(theta) @ RT

    xmin = int(rotated[:, 0].min() * resolution)
    xmax = int(rotated[:, 0].max() * resolution)
    ymin = int(rotated[:, 1].min() * resolution)
    ymax = int(rotated[:, 1].max() * resolution)
    zmax = int(intensity_mask.max())

    xrate = (rotated[:, 0].max() - rotated[:, 0].min())
    yrate = (rotated[:, 1].max() - rotated[:, 1].min())

    shape = (xmax - xmin + 1, ymax - ymin + 1)
    img = np.ma.array(np.ones(shape) * (zmax + 1))
    for i, inp in enumerate(rotated):
        img[int(inp[0] * resolution - xmin), int(inp[1] * resolution - ymin)] = intensity_mask[i]
    img.mask = (img == zmax + 1)
    vertical = do_fourier(
        image=img,
        axis=0,
        rate=xrate
    )
    horizontal = do_fourier(
        image=img,
        axis=1,
        rate=yrate
    )
    return rotated, horizontal / 2, vertical / 2, RT


def ffill_roll(
        arr: np.ndarray,
        fill: int = 0,
        axis: int = 0
) -> np.ndarray:
    mask = arr == None
    replaces = np.roll(arr, 1, axis)
    slicing = tuple(0 if i == axis else slice(None) for i in range(arr.ndim))
    replaces[slicing] = fill
    while np.count_nonzero(mask) > 0:
        arr[mask] = replaces[mask]
        mask = arr == None
        replaces = np.roll(replaces, 1, axis)
    return arr


def do_fourier(
        image: np.ndarray,
        axis: int,
        rate: float
) -> float:
    if axis == 1:
        image = image.T
    primary_range, secondary_range = image.shape[0], image.shape[1]
    peaks = []
    for i in range(secondary_range):
        vec = image.data[:, i]
        vec = np.where(vec > 1, None, vec)
        vec = ffill_roll(vec)
        vec = vec[vec != None]
        if len(vec) == 0:
            continue
        w = np.fft.fft(vec)
        freqs = np.fft.fftfreq(vec.shape[0])
        w = (w.real ** 2 + w.imag ** 2)[freqs > freqs.max() / 25]
        freqs = freqs[freqs > freqs.max() / 25]
        peaks.append(1 / (freqs[np.argmax(w)] * primary_range) * rate)
    return float(np.median(peaks))


def get_lidar_grid(
        image: np.ndarray,
        vertical: float,
        horizontal: float,
        intensity_mask: np.ndarray,
        shape: tuple,
        steps: int = 20
) -> tuple[list[np.ndarray, np.ndarray], float, float]:
    corrs = []
    for sh in np.arange(-horizontal/2,  horizontal/2, horizontal / steps):
        for sv in np.arange(-vertical/3, vertical/3, vertical / steps):
            coeff, diff = calc_corr(
                image=image,
                shift=(sh, sv),
                shape=shape,
                vertical=vertical,
                horizontal=horizontal,
                intensity_mask=intensity_mask,
            )
            corrs.append([sh, sv, coeff, diff])
    corrs = np.array(corrs)
    sh, sv, coeff, diff = corrs[np.argmax(corrs[:, 2])]

    center = [
        (image[:, 0].max() + image[:, 0].min()) / 2 - horizontal * (shape[0]-1) / 2,
        (image[:, 1].max() + image[:, 1].min()) / 2 - vertical * (shape[1]-1) / 2
    ]
    grid_x = np.arange(0, shape[0]) * horizontal + center[0] + sh
    grid_y = np.arange(0, shape[1]) * vertical + center[1] + sv
    grid = [grid_x, grid_y]
    return grid, coeff, diff


def calc_corr(
        image: np.ndarray,
        shift: tuple,
        shape: tuple,
        vertical: float,
        horizontal: float,
        intensity_mask: np.ndarray,
) -> tuple[float, float]:

    center = [
        (image[:, 0].max() + image[:, 0].min()) / 2 - horizontal * (shape[0] - 1) / 2,
        (image[:, 1].max() + image[:, 1].min()) / 2 - vertical * (shape[1] - 1) / 2
    ]
    grid_x = np.arange(0, shape[0]) * horizontal + center[0] + shift[0]
    grid_y = np.arange(0, shape[1]) * vertical + center[1] + shift[1]

    cell_row = np.digitize(image[:, 0], grid_x) - 1
    cell_column = np.digitize(image[:, 1], grid_y) - 1

    cells_id = np.array([i for i in range(shape[0] * shape[1])]).reshape((shape[0], shape[1]))

    ids = cells_id[cell_row, cell_column]
    unq, idx, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    white = np.bincount(idx, weights=intensity_mask)
    black = np.bincount(idx, weights=intensity_mask-1)
    diff = float(np.sum((white + black) / cnt))
    avg = np.bincount(idx, weights=intensity_mask * 2 - 1) / cnt
    coeff = float(np.mean(abs(avg)) * len(unq)/(shape[0] * shape[1]))
    return coeff, diff
