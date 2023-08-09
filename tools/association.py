import shutil

import cv2
import pandas as pd
import numpy as np
import json
import os
from PIL import Image

from tools.utils import NpEncoder

"""
Required input folder structure

folder
|_ lidar
|  |_ pcd
|  |   |_ .pcd
|  |   |_ .pcd
|  |   |_ ....
|  |_ lidar_timestamps.csv
|_ normal_camera
|  |_ images
|  |   |_ .jpg
|  |   |_ .jpg
|  |   |_ ....
|  |_ timestamps.json
|  |_ calib.json
|_ wide_camera
   |_ images
   |   |_ .jpg
   |   |_ .jpg
   |   |_ ....
   |_ timestamps.json
   |_ calib.json
"""


def collect_time_stamps(
        folder_in: str
) -> (pd.Series, pd.Series, pd.Series):
    lid_data = pd.read_csv(folder_in + '/lidar/lidar_timestamps.csv')
    try:
        lid_data = lid_data.rename(columns={'i': 'lidar_frame'})
    except:
        pass
    W, N = [], []
    with open(folder_in + '/normal_camera/timestamps.json') as jsn:
        for line in jsn:
            j = json.loads(line)
            N.append(j)
    with open(folder_in + '/wide_camera/timestamps.json') as jsn:
        for line in jsn:
            j = json.loads(line)
            W.append(j)
    norm_data, wide_data = pd.DataFrame(N), pd.DataFrame(W)
    lidar = lid_data.set_index('lidar_frame')['timestamp']
    lidar.name = 'timestamp'
    normal = norm_data['timeStamp']
    normal.name = 'norm_timestamp'
    wide = wide_data['timeStamp']
    wide.name = 'wide_timestamp'
    return lidar, normal, wide


def associate_frames(
        lidar: pd.Series,
        normal: pd.Series,
        wide: pd.Series,
        n_lag: float = 0.0,
        w_lag: float = 0.0
) -> pd.DataFrame:
    normal += n_lag
    wide += w_lag
    mins, maxs = np.array([lidar.min(), normal.min(), wide.min()]), np.array([lidar.max(), normal.max(), wide.max()])
    Min, Max = mins.max(), maxs.min()
    lidar = lidar.loc[(lidar >= Min) & (lidar <= Max)]
    normal = normal.loc[(normal >= Min) & (normal <= Max)].reset_index()
    normal.rename(columns={'index': 'norm_index'}, inplace=True)
    wide = wide.loc[(wide >= Min) & (wide <= Max)].reset_index()
    wide.rename(columns={'index': 'wide_index'}, inplace=True)
    total = pd.merge_asof(
        lidar.reset_index(),
        normal,
        left_on="timestamp", right_on="norm_timestamp",
        direction="nearest",
        allow_exact_matches=True
    )
    total = pd.merge_asof(
        total,
        wide,
        left_on="timestamp", right_on="wide_timestamp",
        direction="nearest",
        allow_exact_matches=True
    )
    total['n_diff'] = abs(total['timestamp'] - total['norm_timestamp'])
    total['w_diff'] = abs(total['timestamp'] - total['wide_timestamp'])
    return total


def create_output_association_folders(
        folder_out: str
) -> None:
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    if not os.path.exists(f'{folder_out}/lidar'):
        os.mkdir(f'{folder_out}/lidar')

    if not os.path.exists(f'{folder_out}/normal_camera'):
        os.mkdir(f'{folder_out}/normal_camera')
    if not os.path.exists(f'{folder_out}/normal_camera/original'):
        os.mkdir(f'{folder_out}/normal_camera/original')
    if not os.path.exists(f'{folder_out}/normal_camera/undistorted'):
        os.mkdir(f'{folder_out}/normal_camera/undistorted')
    if not os.path.exists(f'{folder_out}/normal_camera/visualization'):
        os.mkdir(f'{folder_out}/normal_camera/visualization')

    if not os.path.exists(f'{folder_out}/wide_camera'):
        os.mkdir(f'{folder_out}/wide_camera')
    if not os.path.exists(f'{folder_out}/wide_camera/original'):
        os.mkdir(f'{folder_out}/wide_camera/original')
    if not os.path.exists(f'{folder_out}/wide_camera/undistorted'):
        os.mkdir(f'{folder_out}/wide_camera/undistorted')
    if not os.path.exists(f'{folder_out}/wide_camera/visualization'):
        os.mkdir(f'{folder_out}/wide_camera/visualization')


def undistort_image(
        image: np.ndarray,
        folder_in: str,
        folder_out: str,
        width: int = 1920,
        height: int = 1080
) -> np.ndarray:
    with open(f'{folder_in}/calib.json') as jsn:
        j = json.load(jsn)
        for data in j['cameraData']:
            if data[0] == 0:
                intrinsic = np.array(data[1]['intrinsicMatrix'])
                distortion = np.array(data[1]['distortionCoeff'])
                coeff_w, coeff_h = data[1]['width'] / width, data[1]['height'] / height
                intrinsic[0, :3] /= coeff_w
                intrinsic[1, :3] /= coeff_h
    new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (width, height), 1, (width, height))
    image = cv2.undistort(image, intrinsic, distortion, None, new_intrinsic)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    matrixes = {
        'intrinsic': intrinsic,
        'distortion': distortion,
        'new_intrinsic': new_intrinsic
    }
    with open(f'{folder_out}/calib.json', 'w') as jsn:
        json.dump(matrixes, jsn, cls=NpEncoder)
    return image


def save_triples(
        association: pd.DataFrame,
        folder_in: str,
        folder_out: str
):
    for index, row in association.iterrows():
        try:
            norm_in_file = f'{folder_in}/normal_camera/images/{str(int(row["norm_index"])).zfill(6)}.jpg'
            norm_out_original_file = f'{folder_out}/normal_camera/original/{str(int(index)).zfill(6)}.jpg'
            norm_out_undistorted_file = f'{folder_out}/normal_camera/undistorted/{str(int(index)).zfill(6)}.jpg'
            wide_in_file = f'{folder_in}/wide_camera/images/{str(int(row["wide_index"])).zfill(6)}.jpg'
            wide_out_original_file = f'{folder_out}/wide_camera/original/{str(int(index)).zfill(6)}.jpg'
            wide_out_undistorted_file = f'{folder_out}/wide_camera/undistorted/{str(int(index)).zfill(6)}.jpg'

            norm_image = np.array(Image.open(norm_in_file))
            wide_image = np.array(Image.open(wide_in_file))

            # Undistort image according to intrinsic matrix
            undistorted_norm_image = undistort_image(
                image=np.array(norm_image),
                folder_in=folder_in + '/normal_camera',
                folder_out=folder_out + '/normal_camera'
            )
            undistorted_wide_image = undistort_image(
                image=np.array(wide_image),
                folder_in=folder_in + '/wide_camera',
                folder_out=folder_out + '/wide_camera'
            )

            cv2.imwrite(norm_out_undistorted_file, undistorted_norm_image)
            cv2.imwrite(wide_out_undistorted_file, undistorted_wide_image)

            shutil.copy(norm_in_file, norm_out_original_file)
            shutil.copy(wide_in_file, wide_out_original_file)

            shutil.copy(f'{folder_in}/lidar/pcd/{str(int(row["lidar_frame"])).zfill(6)}.pcd',
                        f'{folder_out}/lidar/{str(int(index)).zfill(6)}.pcd')
        except FileNotFoundError:
            print('[WARNING] The number of video frames is less than frames in timestamps.json')
            break
    association['index'] = association.index.astype(str)
    association['index'] = association['index'].str.zfill(6)
    association_cut = association.loc[association.index <= index]
    association_cut.to_csv(f'{folder_out}/association.csv', index=False)
