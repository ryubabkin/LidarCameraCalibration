import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def plot_training_curve(
        medians: np.ndarray,
        means: np.ndarray,
        folder_out: str
):
    plt.figure()
    plt.plot(medians, label='median distance')
    plt.plot(means, label='mean distance')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Distance (m)')
    plt.title('Background learning curve:\ndistance between similar background points on consequent frames')
    plt.savefig(f'{folder_out}/learning_curve.png')
    plt.close()


def plot_result(
        img: np.ndarray,
        image_lidar: np.ndarray,
        intensity: np.ndarray,
        folder_out: str,
           idx: int
):
    intensity = intensity[
        (image_lidar[:, 0] >= 0) &
        (image_lidar[:, 0] <= img.shape[1]) &
        (image_lidar[:, 1] >= 0) &
        (image_lidar[:, 1] <= img.shape[0])
    ]

    image_lidar = image_lidar[
        (image_lidar[:, 0] >= 0) &
        (image_lidar[:, 0] <= img.shape[1]) &
        (image_lidar[:, 1] >= 0) &
        (image_lidar[:, 1] <= img.shape[0])
    ]

    plt.figure(figsize=(15, 7))
    plt.scatter(image_lidar[:, 0], image_lidar[:, 1], c=intensity, s=2)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'{folder_out}/{str(idx).zfill(6)}.png', dpi=200)
    plt.close()


def visualize_result(
        folder: str,
        camera_folder: str,
        indx: int,
        intrinsic: np.ndarray,
        distortion: np.ndarray,
        RT_matrix: np.ndarray
):

    pcd_cloud = o3d.t.io.read_point_cloud(f'{folder}/lidar/{str(indx).zfill(6)}.pcd')
    points = np.asarray(o3d.utility.Vector3dVector(pcd_cloud.point.positions.numpy()))
    intensity = np.asarray(pcd_cloud.point.intensity.numpy())

    image = cv2.imread(f'{camera_folder}/original/{str(indx).zfill(6)}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.undistort(image, intrinsic, distortion)

    LI = np.ones([points.shape[0], 4])
    LI[:, :3] = points
    intr = np.eye(4)
    intr[:3, :3] = intrinsic
    image_lidar = (intr @ RT_matrix @ LI.T)
    image_lidar[:2] /= image_lidar[2, :]
    image_lidar = image_lidar.T

    plot_result(
        img=image,
        image_lidar=image_lidar,
        intensity=intensity,
        folder_out=f'{camera_folder}/visualization',
        idx=indx
    )
