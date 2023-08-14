import json
from scipy.spatial.transform import Rotation
import cv2
import numpy as np


def find_chessboard_corners_camera(
        image: np.ndarray,
        cb_size: tuple = (8, 6),
        win_size: int = (11, 11)
) -> np.ndarray:
    gray = cv2.cvtColor(
        src=image.copy(),
        code=cv2.COLOR_BGR2GRAY
    )
    found, corners = cv2.findChessboardCorners(
        image=image,
        patternSize=cb_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    corners_sub_pix = cv2.cornerSubPix(
        image=gray,
        corners=corners,
        winSize=win_size,
        zeroZone=(-1, -1),
        criteria=criteria
    )
    corners_sub_pix = corners_sub_pix.reshape((corners_sub_pix.shape[0], 2))
    return corners_sub_pix


def calculate_RT(
        lidar_markers: np.ndarray,
        image_markers: np.ndarray,
        intrinsic: np.ndarray,
        distortion: np.ndarray,
        hfov: float,
        vfov: float,
        image_size: tuple = (1920, 1080),
        reprojection_error: float = 20,
) -> tuple[np.ndarray, float, list]:
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=lidar_markers,
        imagePoints=image_markers,
        cameraMatrix=intrinsic,
        distCoeffs=distortion,
        iterationsCount=1000,
        reprojectionError=reprojection_error,
        confidence=0.9999,
    )
    coefficient = len(inliers) / lidar_markers.shape[0]
    R, _ = cv2.Rodrigues(rvec)
    RT = np.eye(4)
    RT[:3, :4] = np.hstack((R, tvec))

    # check the accuracy
    LI = np.ones([lidar_markers.shape[0], 4])
    LI[:, :3] = lidar_markers
    intr = np.eye(4)
    intr[:3, :3] = intrinsic
    image_lidar_markers = (intr @ RT @ LI.T)
    image_lidar_markers[:2] /= image_lidar_markers[2, :]
    image_lidar_markers = image_lidar_markers.T

    error_pxl_X = image_markers[:, 0] - image_lidar_markers[:, 0]
    error_pxl_Y = image_markers[:, 1] - image_lidar_markers[:, 1]
    error_deg_X = error_pxl_X * hfov / image_size[0]
    error_deg_Y = error_pxl_Y * vfov / image_size[1]
    error_pxl = np.sqrt(error_pxl_X ** 2 + error_pxl_Y ** 2)
    error_deg = np.sqrt(error_deg_X ** 2 + error_deg_Y ** 2)
    rmse_pxl = float(np.sqrt(np.mean(error_pxl ** 2)))
    mae_pxl = float(np.mean(np.abs(error_pxl)))
    rmse_deg = float(np.sqrt(np.mean(error_deg ** 2)))
    mae_deg = float(np.mean(np.abs(error_deg)))

    return RT, coefficient, [rmse_pxl, mae_pxl, rmse_deg, mae_deg]


def prepare_params_json(
        folder: str,
        camera_params: dict,
        RT_matrix: np.ndarray,
        error: list,
        points: float,
        distance: float,
        elevation: float = 0.0,
        angles: tuple = (0, 0, 0),
        resolution: tuple = (1920, 1080),
):
    RT_matrix = np.linalg.inv(RT_matrix)
    quaternion = Rotation.from_matrix(RT_matrix[:3, :3]).as_quat()
    translation = RT_matrix[:3, 3]
    params = {
        "lidar_extrinsic": {
            "rotation_x": quaternion[0],
            "rotation_y": quaternion[1],
            "rotation_z": quaternion[2],
            "rotation_w": quaternion[3],
            "x": translation[0],
            "y": translation[1],
            "z": translation[2],
            "f_x": camera_params['new_intrinsic'][0][0],
            "f_y": camera_params['new_intrinsic'][1][1],
            "c_x": camera_params['new_intrinsic'][0][2],
            "c_y": camera_params['new_intrinsic'][1][2]
        },
        "intrinsic": camera_params['intrinsic'],
        "distortion": camera_params['distortion'],
        "elevation": elevation,
        "angles": list(angles),
        "resolution": list(resolution),
        "accuracy": {
            "RMSE_pixels": error[0],
            "MAE_pixels": error[1],
            "RMSE_degrees": error[2],
            "MAE_degrees": error[3],
            "avg_points_per_cell": points,
            "avg_distance_to_chessboard": distance
        }
    }
    with open(f'{folder}/calib.json', 'w') as f:
        json.dump(params, f, indent=4)
        f.close()
