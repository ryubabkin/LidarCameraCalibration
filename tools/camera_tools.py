import json

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
        distortion: np.ndarray
) -> np.ndarray:
    _, rvec, tvec, _ = cv2.solvePnPRansac(
        objectPoints=lidar_markers,
        imagePoints=image_markers,
        cameraMatrix=intrinsic,
        distCoeffs=distortion
    )
    R, _ = cv2.Rodrigues(rvec)
    RT = np.hstack((R, tvec))
    return RT


def prepare_params_json(
        folder: str,
        camera_params: dict,
        RT_matrix: np.ndarray,
        elevation: float = 0.0,
        angles: tuple = (0, 0, 0),
        resolution: tuple = (1920, 1080),
):
    params = {
        "lidar_extrinsic": {
            "rotation_x": -0.499039,
            "rotation_y": 0.518256,
            "rotation_z": -0.491861,
            "rotation_w": 0.490350,
            "x": 0.07509793586623727,
            "y": 1.1985334811054549,
            "z": 0.019436941577036237,
            "f_x": camera_params['new_intrinsic'][0][0],
            "f_y": camera_params['new_intrinsic'][1][1],
            "c_x": camera_params['new_intrinsic'][0][2],
            "c_y": camera_params['new_intrinsic'][1][2]
        },

        "intrinsic": camera_params['intrinsic'],
        "distortion": camera_params['distortion'],
        "elevation": elevation,
        "angles": list(angles),
        "resolution": list(resolution)
    }
    with open(f'{folder}/calib.json', 'w') as f:
        json.dump(params, f, indent=4)
        f.close()
