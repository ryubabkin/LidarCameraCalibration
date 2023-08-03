import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely as shp

def plot_background_points(
        points: np.ndarray,
        background_points: np.ndarray,
        intensity: np.ndarray,
        indx: int,
    folder_out: str
):
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    try:
        ax1.scatter(background_points[:, 1], background_points[:, 0], c='lightgray', alpha=0.2)
        ax2.scatter(background_points[:, 1], background_points[:, 2], c='lightgray', alpha=0.2)
        ax3.scatter(background_points[:, 0], background_points[:, 2], c='lightgray', alpha=0.2)
    except:
        pass
    ax1.scatter(points[:, 1], points[:, 0], c=intensity, alpha=0.5)
    ax2.scatter(points[:, 1], points[:, 2], c=intensity, alpha=0.5)
    ax3.scatter(points[:, 0], points[:, 2], c=intensity, alpha=0.5)

    ax1.invert_xaxis()
    ax1.set_title(indx)
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'{folder_out}/lidar/{indx}.png', dpi=120)
    plt.close()


def plot_training_curve(
        medians: np.ndarray,
        means: np.ndarray,
        folder_out: str
):
    plt.figure()
    plt.plot(medians)
    plt.plot(means)
    plt.savefig(f'{folder_out}/TRAINING_CURVE.png')
    plt.show()


def plot_lidar_chessboard(
        chessboard: np.ndarray,
        markers: np.ndarray,
        indx: int,
        folder_out: str
):
    blacks = chessboard[chessboard[:, 3] == 0][:, :3]
    whites = chessboard[chessboard[:, 3] == 1][:, :3]

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.scatter(blacks[:, 0], blacks[:, 1], s=1, c='red')
    ax.scatter(whites[:, 0], whites[:, 1], s=1, c='lightblue')
    ax.scatter(markers[:, 0], markers[:, 1], marker="$\u25EF$", edgecolor='k', s=100)

    ax.invert_xaxis()
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{folder_out}/lidar/{indx}.png', dpi=120)
    plt.close()


def plot_cb_corners_camera(
        image: np.ndarray,
        corners: np.ndarray,
        folder_out: str,
        indx: int
):
    gray = cv2.cvtColor(
        src=image,
        code=cv2.COLOR_GRAY2BGR
    )
    MIN, MAX = corners.min(axis=0), corners.max(axis=0)
    plt.figure()
    plt.imshow(gray)
    plt.scatter(corners[:, 0], corners[:, 1], marker="$\u25EF$", edgecolor='lime', s=50)
    plt.xlim(MIN[0] - 100, MAX[0] + 100)
    plt.ylim(MAX[1] + 100, MIN[1] - 100)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{folder_out}/images/{indx}.png', dpi=120)
    # plt.show()
    plt.close()
