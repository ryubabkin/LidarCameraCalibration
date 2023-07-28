from __future__ import annotations

import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation
import cv2
from skimage.transform import ProjectiveTransform
import matplotlib.path as mplPath
import shapely as shp
import os


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
) -> (np.ndarray, np.ndarray):
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


def matrix_to_angles(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 4:
        matrix = matrix[:3, :3]
    return Rotation.from_matrix(matrix).as_euler('xyz', degrees=True)


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion).as_matrix()


def angles_to_matrix(angles: list | np.ndarray | tuple, dim: int = 4) -> np.ndarray:
    mat = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    if dim == 4:
        matrix = np.eye(4)
        matrix[:3, :3] = mat
        return matrix
    else:
        matrix = mat
    return matrix


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


def get_best_4gram(
        contour: np.ndarray,
        sort=True
) -> np.ndarray:
    """
    A sequence of getting best-fit quadrilateral as an average between the best inner and best outer quadrilaterals
    """
    if sort:
        contour = sort_contour(points=contour)
    inner = get_inner_4gram(contour=contour)
    outer = get_outer_4gram(contour=contour, inner=inner)
    inner = sort_4gram(pts=inner)
    outer = sort_4gram(pts=outer)
    interm = (inner + outer) / 2
    return interm


def select_segment_on_lidar(
        cloud: np.ndarray,
        segment: np.ndarray,
        camera
) -> (np.ndarray, np.ndarray):
    """
    Selects lidar points that are surrounded by a contour on image
    """
    img_pcd, _ = camera.lidar_to_image(cloud, crop=False)
    # img_pcd = camera.undistort_points(img_pcd)
    polygon = mplPath.Path(segment)
    mask = polygon.contains_points(img_pcd.T.astype(int))
    return cloud[:, mask], mask


def IoU(
        a: np.ndarray,
        b: np.ndarray
) -> float:
    """
    Calculate intersection over union (2d)
    """
    a_poly, b_poly = shp.Polygon(a), shp.Polygon(b)
    intersection = a_poly.intersection(b_poly).area
    union = a_poly.union(b_poly).area
    iou = intersection / union
    return iou


def get_RT(
        rot_angles: tuple,  # (pitch, roll, yaw)
        trans_shifts: tuple,  # ( --, / , | )
        inv: bool
) -> np.ndarray:
    """
    create Rotation-translation matrix (RT). If inv==True - calculates an inverted one
    """
    RT = angles_to_matrix(angles=rot_angles)
    RT[:3, 3] = trans_shifts
    RT = RT @ Rz(90, f=True)
    if inv:
        RT = np.linalg.inv(RT)
    return RT


def read_input_pair(
        image_folder,
        lidar_folder,
        frame_id
) -> (np.dnarray, o3d.geometry.PointCloud, np.ndarray):
    image_name = f'{frame_id}.jpg'
    pcd_files = os.listdir(lidar_folder)
    pcd_name = [file for file in pcd_files if file.split('_')[1].split('.')[0] == frame_id][0]

    # image = mpimg.imread(os.path.join(image_folder, image_name))
    image = None
    pcd = o3d.t.io.read_point_cloud(os.path.join(lidar_folder, pcd_name))
    intensity = pcd.point.intensity.numpy()
    pcd_cloud = o3d.geometry.PointCloud()
    pcd_cloud.points = o3d.utility.Vector3dVector(pcd.point.positions.numpy())
    return image, pcd_cloud, intensity


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
