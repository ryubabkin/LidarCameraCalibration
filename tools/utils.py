from __future__ import annotations

import time
import traceback

import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation
import cv2
from skimage.transform import ProjectiveTransform
import matplotlib.path as mplPath
import shapely as shp
import os


class LogsException(object):
    def __init__(self):
        self.pid = os.getpid()

    @staticmethod
    def time():
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def error(self, text: str):
        print(f"[{self.time()}] [{self.pid}] [ERROR] {text}")
        traceback.print_exc()

    def done(self, text: str):
        print(f"[{self.time()}] [{self.pid}] [DONE] {text}")

    def start(self, text: str):
        print(f"[{self.time()}] [{self.pid}] [START] {text}")

    def info(self, text: str):
        print(f"[{self.time()}] [{self.pid}] [INFO] {text}")

    def warn(self, text: str):
        print(f"[{self.time()}] [{self.pid}] [WARNING] {text}")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)


def from_pcd(
        file_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read pcd file
    """
    pcd = o3d.t.io.read_point_cloud(file_name)
    points = pcd.point.positions.numpy()
    intensities = pcd.point.intensity.numpy()
    return points.T, intensities.flatten()


def Rx(
        theta: float,
        f: bool = False
) -> np.ndarray:
    """
    Rotate around x-axis
    """
    theta = np.pi * theta / 180
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    if f:
        r = np.eye(4)
        r[:3, :3] = rot
        return r
    else:
        return rot


def Ry(
        theta: float,
        f: bool = False
) -> np.ndarray:
    """
    Rotate around y-axis
    """
    theta = np.pi * theta / 180
    rot = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    if f:
        r = np.eye(4)
        r[:3, :3] = rot
        return r
    else:
        return rot


def Rz(
        theta: float,
        f: bool = False
) -> np.ndarray:
    """
    Rotate around z-axis
    """
    theta = np.pi * theta / 180
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    if f:
        r = np.eye(4)
        r[:3, :3] = rot
        return r
    else:
        return rot


def sort_contour(
        points: np.ndarray  # Nx2
) -> np.ndarray:
    """
    Counter-clockwise sort of points of the contour
    """
    if points.shape[0] == 2:
        points = points.T
    center_x, center_y = points[:, 0].mean(), points[:, 1].mean()
    angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)
    indices = np.argsort(angles)
    sorted_points = points[indices, :]
    return sorted_points


def get_inner_4gram(
        contour: np.ndarray
) -> np.ndarray:
    """
    Search for the best-fit inner quadrilateral
    """
    hull = cv2.convexHull(contour)
    hull = hull.reshape(hull.shape[0], 2)
    epsilon = 0.05 * cv2.arcLength(curve=hull, closed=True)
    inner = cv2.approxPolyDP(curve=hull, epsilon=epsilon, closed=True)
    if inner.shape[1] == 1:
        inner = np.hstack([inner[:, :, 0], inner[:, :, 1]])
    inner = sort_4gram(inner)
    return inner


def get_outer_4gram(
        contour: np.ndarray,
        inner: np.ndarray
) -> np.ndarray:
    """
    Search for the best-fit outer quadrilateral
    """
    Transform = ProjectiveTransform()
    sample = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
    if not Transform.estimate(src=inner, dst=sample):
        raise Exception("estimate failed")
    contour_squared = Transform(contour.reshape(contour.shape[0], 2)).reshape(contour.shape[0], 1, 2).astype(np.float32)
    mar = cv2.minAreaRect(points=contour_squared)
    box = cv2.boxPoints(box=mar)
    outer = Transform.inverse(coords=box)
    return outer


def sort_4gram(
        pts: np.ndarray
) -> np.ndarray:
    """
    Sort points counter-clockwise and select the best quadrilateral
    """
    pts_copy = pts.copy()
    pts4 = []
    s = np.sum(pts_copy, axis=1).flatten()
    pts4.append(pts[np.argmin(s)])
    pts_copy = np.delete(pts_copy, np.argmin(s), axis=0)

    diff = np.diff(pts_copy, axis=1).flatten()
    pts4.append(pts_copy[np.argmax(diff)])
    pts_copy = np.delete(pts_copy, np.argmax(diff), axis=0)

    s = np.sum(pts_copy, axis=1).flatten()
    pts4.append(pts_copy[np.argmax(s)])
    pts_copy = np.delete(pts_copy, np.argmax(s), axis=0)

    diff = np.diff(pts_copy, axis=1).flatten()
    pts4.append(pts_copy[np.argmin(diff)])

    pts4 = np.array(pts4)
    center = pts4.mean(axis=0).reshape(1, -1)
    angles = np.arctan2((pts4 - center)[:, 0], (pts4 - center)[:, 1])
    return pts4[np.argsort(angles)]


def get_segmentation_box(contour: np.ndarray) -> np.ndarray:
    """
    Get boundary box of a contour
    """
    polygon = shp.Polygon(contour)
    box = np.array(polygon.envelope.exterior.xy).T
    return sort_4gram(pts=box)


def choose_best_plane(
        points: np.ndarray,
        inlier_threshold: float = 0.02
) -> (np.ndarray, np.ndarray, float):
    """
    chooses the best plane using RANSAC algorithm
    """
    mask = np.zeros(len(points), dtype=bool)
    cloud = o3d.t.geometry.PointCloud()
    cloud.point.positions = o3d.core.Tensor(points)
    best_eq, inliers = cloud.segment_plane(
        distance_threshold=inlier_threshold,
        ransac_n=3,
        probability=0.85
    )
    # best_eq = fit_plane(points[inliers.numpy()])
    best_inliers = inliers.numpy()
    mask[best_inliers] = True
    confidence = len(best_inliers) / len(mask)
    return mask, best_eq.numpy(), confidence


def fit_plane(
        points: np.ndarray
) -> np.ndarray:
    """
    Method calculates the equation of the plane
    """
    XY1 = np.ones((points.shape[0], 3))
    XY1[:, 0], XY1[:, 1] = points[:, 0], points[:, 1]
    Z = points[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(XY1, Z, rcond=None)
    normal = np.array([a, b, -1])
    normal = normal / np.linalg.norm(normal)
    distance = -normal.dot(points.mean(axis=0))
    plane = np.hstack([normal, distance])
    return plane


def vec2vec(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
