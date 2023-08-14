import json
import sys
import os
from tools.association import create_output_association_folders
from tools.flows import Calibration_Flow, Association_Flow
from tools.lidar_tools import background_detection
from tools.utils import LogsException

LogExc = LogsException()


def run(
        params: dict,
        extract: bool = True,
        calibrate: bool = True
):
    LogExc.start("====== START ======")
    LogExc.info(params)
    folder_in = params['folders']['folder_in']
    folder_out = params['folders']['folder_out']

    create_output_association_folders(folder_out=folder_out)
    association = Association_Flow(
        folder_in=folder_in,
        folder_out=folder_out,
        n_lag_seconds=params['lag_seconds_normal'],
        w_lag_seconds=params['lag_seconds_wide'],
        extract=extract
    )

    if calibrate:
        LogExc.start("Lidar background detection...")
        background_pcd, stop_id = background_detection(
            association=association,
            folder=folder_out,
            max_distance=params['max_distance'],
            median_distance_stop=params['median_distance_stop'],
        )
        LogExc.done("Lidar background detection")
        if os.path.isdir(f'{folder_in}/normal_camera/') and ('norm_timestamp' in association.columns):
            LogExc.start('Normal camera calibration')
            Calibration_Flow(
                camera_folder=f'{folder_out}/normal_camera',
                folder=folder_out,
                association=association,
                background_pcd=background_pcd,
                stop_id=stop_id,
                hfov=params['normal_camera_parameters']['hfov'],
                vfov=params['normal_camera_parameters']['vfov'],
                elevation=params['normal_camera_parameters']['elevation'],
                angles=params['normal_camera_parameters']['angles'],
                max_distance=params['max_distance'],
                min_points_in_chessboard=params['min_points_in_chessboard'],
                min_delta=params['min_delta'],
                cluster_threshold=params['cluster_threshold'],
                min_points_in_cluster=params['min_points_in_cluster'],
                cb_cells=params['chessboard_cells'],
                n_points_interpolate=params['n_points_interpolate'],
                period_resolution=params['period_resolution'],
                grid_steps=params['grid_steps'],
                grid_threshold=params['grid_threshold'],
                reprojection_error=params['reprojection_error']
            )
            LogExc.done("Normal camera calibration")
        else:
            LogExc.warn("Normal camera folder not found")
        if os.path.isdir(f'{folder_in}/wide_camera/') and ('wide_timestamp' in association.columns):
            LogExc.start('Wide camera calibration')
            Calibration_Flow(
                camera_folder=f'{folder_out}/wide_camera',
                folder=folder_out,
                association=association,
                background_pcd=background_pcd,
                stop_id=stop_id,
                hfov=params['wide_camera_parameters']['hfov'],
                vfov=params['wide_camera_parameters']['vfov'],
                elevation=params['wide_camera_parameters']['elevation'],
                angles=params['wide_camera_parameters']['angles'],
                max_distance=params['max_distance'],
                min_points_in_chessboard=params['min_points_in_chessboard'],
                min_delta=params['min_delta'],
                cluster_threshold=params['cluster_threshold'],
                min_points_in_cluster=params['min_points_in_cluster'],
                cb_cells=params['chessboard_cells'],
                n_points_interpolate=params['n_points_interpolate'],
                period_resolution=params['period_resolution'],
                grid_steps=params['grid_steps'],
                grid_threshold=params['grid_threshold'],
                reprojection_error=params['reprojection_error']
            )
            LogExc.done("Wide camera calibration")
        else:
            LogExc.warn("Wide camera folder not found")
    LogExc.done("====== DONE ======")

# %%


if __name__ == '__main__':
    settings_file = sys.argv[1]
    extract = bool(int(sys.argv[2]))
    calibrate = bool(int(sys.argv[3]))
    print(extract, calibrate)
    # settings_file = '/Users/brom/Laboratory/GlobalLogic/MEAA/LidarCameraCalibration/settings.json'
    # extract = False
    # calibrate = True

    with open(settings_file, 'r') as file:
        params = json.load(file)
        file.close()
    run(
        params=params,
        extract=extract,
        calibrate=calibrate
    )
    sys.exit(0)
