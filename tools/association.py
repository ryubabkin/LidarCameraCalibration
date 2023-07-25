import shutil

import cv2
import pandas as pd
import numpy as np
import json
import os
from PIL import Image

from utils import NpEncoder

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
        folder_in: str,
        zero_index: bool = True
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
    norm_data = norm_data.rename(columns={'frameNumber': 'normal_frame'})
    wide_data = wide_data.rename(columns={'frameNumber': 'wide_frame'})
    lidar = lid_data.set_index('lidar_frame')['timestamp']
    lidar.name = 'timestamp'
    normal = norm_data.set_index('normal_frame')['timeStamp']
    normal.name = 'norm_timestamp'
    wide = wide_data.set_index('wide_frame')['timeStamp']
    wide.name = 'wide_timestamp'
    if zero_index:
        normal.index = normal.index - normal.index.min()
        wide.index = wide.index - wide.index.min()
    return lidar, normal, wide


def associate_frames(
        lidar: pd.Series,
        normal: pd.Series,
        wide: pd.Series,
        n_lag: int = 0,
        w_lag: int = 0
) -> pd.DataFrame:
    normal += n_lag
    wide += w_lag
    mins, maxs = np.array([lidar.min(), normal.min(), wide.min()]), np.array([lidar.max(), normal.max(), wide.max()])
    Min, Max = mins.max(), maxs.min()
    lidar = lidar.loc[(lidar >= Min) & (lidar <= Max)]
    normal = normal.loc[(normal >= Min) & (normal <= Max)]
    wide = wide.loc[(wide >= Min) & (wide <= Max)]

    total = pd.merge_asof(
        lidar.reset_index(),
        normal.reset_index(),
        left_on="timestamp", right_on="norm_timestamp",
        direction="nearest",
        allow_exact_matches=True
    )
    total = pd.merge_asof(
        total.reset_index(),
        wide.reset_index(),
        left_on="timestamp", right_on="wide_timestamp",
        direction="nearest",
        allow_exact_matches=True
    )
    total['n_diff'] = abs(total['timestamp'] - total['norm_timestamp'])
    total['w_diff'] = abs(total['timestamp'] - total['wide_timestamp'])
    return total


def create_output_folders(
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

    if not os.path.exists(f'{folder_out}/wide_camera'):
        os.mkdir(f'{folder_out}/wide_camera')
    if not os.path.exists(f'{folder_out}/wide_camera/original'):
        os.mkdir(f'{folder_out}/wide_camera/original')
    if not os.path.exists(f'{folder_out}/wide_camera/undistorted'):
        os.mkdir(f'{folder_out}/wide_camera/undistorted')


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
    create_output_folders(folder_out)
    for index, row in association.iterrows():
        try:
            norm_in_file = f'{folder_in}/normal_camera/images/{str(int(row["normal_frame"])).zfill(6)}.jpg'
            norm_out_original_file = f'{folder_out}/normal_camera/original/{str(int(row["index"])).zfill(6)}.jpg'
            norm_out_undistorted_file = f'{folder_out}/normal_camera/undistorted/{str(int(row["index"])).zfill(6)}.jpg'
            wide_in_file = f'{folder_in}/wide_camera/images/{str(int(row["wide_frame"])).zfill(6)}.jpg'
            wide_out_original_file = f'{folder_out}/wide_camera/original/{str(int(row["index"])).zfill(6)}.jpg'
            wide_out_undistorted_file = f'{folder_out}/wide_camera/undistorted/{str(int(row["index"])).zfill(6)}.jpg'

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
                        f'{folder_out}/lidar/{str(int(row["index"])).zfill(6)}.pcd')
        except FileNotFoundError:
            print('[WARNING] The number of video frames is less than frames in timestamps.json')
            break
    association_cut = association.loc[association['index'] <= row["index"]]
    association_cut.to_csv(f'{folder_out}/association.csv', index=False)
    print('[DONE] Association is finished')


def flow(
        folder_in: str,
        folder_out: str
):
    L, N, W = collect_time_stamps(
        folder_in=folder_in,
    )
    associated_frames = associate_frames(
        lidar=L,
        normal=N,
        wide=W,
        n_lag=0,
        w_lag=0
    )
    save_triples(
        association=associated_frames,
        folder_in=folder_in,
        folder_out=folder_out
    )


if __name__ == '__main__':
    folder_in = '/Users/brom/Laboratory/GlobalLogic/MEAA/LidarCameraCalibration/data/first'
    folder_out = '/Users/brom/Laboratory/GlobalLogic/MEAA/LidarCameraCalibration/data/first/RESULT/'
    flow(folder_in=folder_in, folder_out=folder_out)

