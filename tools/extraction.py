from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import open3d as o3d
import struct
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image


"""
Required input folder structure

folder
|_ lidar
|  |_ rosbag2_YYYY_MM_DD-HH_MM_SS
|_ normal_camera
|  |_ color.mjpeg
|  |_ timestamps.json
|  |_ calib.json
|_ wide_camera
   |_ color.mjpeg
   |_ timestamps.json
   |_ calib.json
"""


def create_output_folders(
        folder_out: str
) -> None:
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    if not os.path.exists(f'{folder_out}/lidar'):
        os.mkdir(f'{folder_out}/lidar')
    if not os.path.exists(f'{folder_out}/lidar/pcd'):
        os.mkdir(f'{folder_out}/lidar/pcd')

    if not os.path.exists(f'{folder_out}/normal_camera'):
        os.mkdir(f'{folder_out}/normal_camera')
    if not os.path.exists(f'{folder_out}/normal_camera/images'):
        os.mkdir(f'{folder_out}/normal_camera/images')

    if not os.path.exists(f'{folder_out}/wide_camera'):
        os.mkdir(f'{folder_out}/wide_camera')
    if not os.path.exists(f'{folder_out}/wide_camera/images'):
        os.mkdir(f'{folder_out}/wide_camera/images')



def float_from_bytes(
        bytes0: bytes,
        big_endian: bool = False
) -> tuple:
    fmt = '>f' if big_endian else '<f'
    flt = struct.unpack(fmt, bytes0)[0]
    return flt


def import_lidar_rawdata(
        raw_data: bytes,
        connection
) -> pd.DataFrame:
    msg = deserialize_cdr(raw_data, connection.msgtype)
    frame = []
    for r in range(int(msg.row_step / msg.point_step)):
        x = float_from_bytes(msg.data[msg.point_step * r + 0:msg.point_step * r + 4])
        y = float_from_bytes(msg.data[msg.point_step * r + 4:msg.point_step * r + 8])
        z = float_from_bytes(msg.data[msg.point_step * r + 8:msg.point_step * r + 12])
        i = float_from_bytes(msg.data[msg.point_step * r + 12:msg.point_step * r + 16])
        frame.append([x, y, z, i])
    frame = np.array(frame)
    frame = frame[~np.all(frame == 0, axis=1)]
    return pd.DataFrame(frame)


def do_rescale(
        image: np.ndarray,
        scale: int
) -> np.ndarray:
    pImage = Image.fromarray(image)
    rescaled = np.array(pImage.resize(
        size=(int(image.shape[1] * scale), int(image.shape[0] * scale)),
        resample=Image.BICUBIC)
    )
    return rescaled


def extract_video_frames(
        filename_in: str,
        output_folder: str,
        prefix: str = '',
        start_time_sec: int = 0,
        end_time_sec: int = None
) -> None:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    vidcap = cv2.VideoCapture(filename_in)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(output_folder + prefix + '%06d.jpg' % count, image)
        success, image = vidcap.read()
        if end_time_sec is not None and end_time_sec < 1000000:
            current_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
            if current_time > 1000 * end_time_sec: success = False
        count += 1
    return


def extract_rosbag_frames(
        rosbag_folder: str,
        output_folder: str,
        timestamps_file: str,
        prefix: str = '',
) -> None:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    timestamps = pd.DataFrame()
    with Reader(rosbag_folder) as reader:
        connections = [x for x in reader.connections if x.topic == '/livox/lidar']
        i = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            df_frame = import_lidar_rawdata(rawdata, connection)
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(df_frame.values[:, :3].astype(np.float32))
            pcd.point["intensity"] = o3d.core.Tensor(df_frame.values[:, 3].reshape((-1, 1)).astype(np.float32))
            o3d.t.io.write_point_cloud(output_folder + prefix + '/%06d.pcd' % i, pcd)
            timestamp = {'lidar_frame': int(i), 'timestamp': timestamp / 1000_000_000}
            timestamps = pd.concat([timestamps, pd.DataFrame([timestamp])], ignore_index=True)
            i += 1

    timestamps['lidar_frame'] = timestamps['lidar_frame'].astype(int)
    timestamps.to_csv(timestamps_file, index=False)
    return


def extract(
    folder_in: str,
) -> None:
    create_output_folders(
        folder_out=folder_in
    )
    extract_video_frames(
        filename_in=f'{folder_in}/normal_camera/color.mjpeg',
        output_folder=f'{folder_in}/normal_camera/images/',
        prefix='',
        start_time_sec=0,
        end_time_sec=None
    )
    extract_video_frames(
        filename_in=f'{folder_in}/wide_camera/color.mjpeg',
        output_folder=f'{folder_in}/wide_camera/images/',
        prefix='',
        start_time_sec=0,
        end_time_sec=None
    )
    rosbag_folder = [folder for folder in os.listdir(f'{folder_in}/lidar/') if folder.startswith("rosbag2")][0]
    extract_rosbag_frames(
        rosbag_folder=f'{folder_in}/lidar/{rosbag_folder}',
        output_folder=f'{folder_in}/lidar/pcd/',
        timestamps_file=f'{folder_in}/lidar/lidar_timestamps.csv',
        prefix=''
    )
    return

extract(
    folder_in='/Users/brom/Laboratory/GlobalLogic/MEAA/DATA/updated_rover_setup_v2_GL/refrigerated_maneuvers',
)