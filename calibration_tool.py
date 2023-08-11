import json
import sys
import os

from tools.association import create_output_association_folders
from tools.flows import Calibration_Flow, Association_Flow
from tools.lidar_tools import background_detection


def run(
        settings_file: str,
        folder_in: str,
        folder_out: str,
        calibrate: bool = True,
        extract: bool = True
):
    with open(settings_file, 'r') as file:
        args_list = json.load(file)
        file.close()
    print('[START] Extracting and Association')
    create_output_association_folders(folder_out=folder_out)
    association = Association_Flow(
        folder_in=folder_in,
        folder_out=folder_out,
        n_lag_seconds=args_list['lag_seconds_normal'],
        w_lag_seconds=args_list['lag_seconds_wide'],
        extract=extract
    )

    if calibrate:

        print("Lidar background detection...")
        background_pcd, stop_id = background_detection(
            association=association,
            folder=folder_out,
            max_distance=args_list['max_distance'],
            median_distance_stop=args_list['median_distance_stop'],
        )
        print("[DONE] Lidar background detection")
        if os.path.isdir(f'{folder_in}/normal_camera/'):
            print('[START] Normal camera calibration')
            Calibration_Flow(
                camera_folder=f'{folder_out}/normal_camera',
                folder=folder_out,
                association=association,
                background_pcd=background_pcd,
                stop_id=stop_id,
                elevation=args_list['normal_camera_parameters']['elevation'],
                angles=args_list['normal_camera_parameters']['angles'],
                max_distance=args_list['max_distance'],
                min_points_in_chessboard=args_list['min_points_in_chessboard'],
                min_delta=args_list['min_delta'],
                cluster_threshold=args_list['cluster_threshold'],
                min_points_in_cluster=args_list['min_points_in_cluster'],
                cb_cells=args_list['chessboard_cells'],
                n_points_interpolate=args_list['n_points_interpolate'],
                period_resolution=args_list['period_resolution'],
                grid_steps=args_list['grid_steps'],
                grid_threshold=args_list['grid_threshold'],
                reprojection_error=args_list['reprojection_error']
            )
            print("[DONE] Normal camera calibration")
        else:
            print("[WARNING] Normal camera folder not found")
        if os.path.isdir(f'{folder_in}/wide_camera/'):
            print('[START] Wide camera calibration')
            Calibration_Flow(
                camera_folder=f'{folder_out}/wide_camera',
                folder=folder_out,
                association=association,
                background_pcd=background_pcd,
                stop_id=stop_id,
                elevation=args_list['wide_camera_parameters']['elevation'],
                angles=args_list['wide_camera_parameters']['angles'],
                max_distance=args_list['max_distance'],
                min_points_in_chessboard=args_list['min_points_in_chessboard'],
                min_delta=args_list['min_delta'],
                cluster_threshold=args_list['cluster_threshold'],
                min_points_in_cluster=args_list['min_points_in_cluster'],
                cb_cells=args_list['chessboard_cells'],
                n_points_interpolate=args_list['n_points_interpolate'],
                period_resolution=args_list['period_resolution'],
                grid_steps=args_list['grid_steps'],
                grid_threshold=args_list['grid_threshold'],
                reprojection_error=args_list['reprojection_error']
            )
            print("[DONE] Wide camera calibration")
        else:
            print("[WARNING] Wide camera folder not found")
    print("====== DONE ======")

# %%


if __name__ == '__main__':
    # if len(sys.argv) < 6:
    #     print("[ERROR] Wrong number of input parameters")
    #     sys.exit(1)
    # if sys.argv[4] not in [0, 1]:
    #     print("[ERROR] Wrong value for extract parameter")
    #     sys.exit(1)
    # if sys.argv[5] not in [0, 1]:
    #     print("[ERROR] Wrong value for calibrate parameter")
    #     sys.exit(1)
    # settings_file = sys.argv[1]
    # folder_in = sys.argv[2]
    # folder_out = sys.argv[3]
    # extract = bool(sys.argv[4])
    # calibrate = bool(sys.argv[5])

    folder_in = '/Users/brom/Laboratory/GlobalLogic/MEAA/LidarCameraCalibration/data/second'
    folder_out = f'{folder_in}/output'
    settings_file = './settings.json'
    extract = False
    calibrate = True
    run(
        settings_file=settings_file,
        folder_in=folder_in,
        folder_out=folder_out,
        calibrate=calibrate,
        extract=extract
    )
    sys.exit(0)
