import cv2
import numpy as np

from tools.plotting import plot_cb_corners_camera


def find_chessboard_corners_camera(
        image: np.ndarray,
        folder_out: str,
        cb_size: tuple = (8, 6),
        win_size: int = (11, 11),
        idx: int = 0,
        plot: bool = True
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
    if plot:
        plot_cb_corners_camera(
            image=gray.copy(),
            corners=corners_sub_pix,
            folder_out=folder_out,
            indx=idx,
        )
    return corners_sub_pix


def calculate_RT(
        lidar_markers: np.ndarray,
        image_markers: np.ndarray,
        intrinsic: np.ndarray,
        distortion: np.ndarray
) -> np.ndarray:
    _, rvec, tvec = cv2.solvePnP(
        objectPoints=lidar_markers,
        imagePoints=image_markers,
        cameraMatrix=intrinsic,
        distCoeffs=distortion
    )
    R, _ = cv2.Rodrigues(rvec)
    RT = np.hstack((R, tvec))
    return RT
