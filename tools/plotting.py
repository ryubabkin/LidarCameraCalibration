import matplotlib.pyplot as plt
import numpy as np


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
    plt.savefig(f'{folder_out}/{indx}.png', dpi=120)
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


def plot_interpolated_lidar_cb(
        coordinates: np.ndarray,
        mask: np.ndarray,
        background_points: np.ndarray,
        indx: int,
        folder_out: str
):
    plt.figure(figsize=(10,5))
    # ax1 = plt.subplot(311)
    ax2 = plt.subplot(111)
    # ax3 = plt.subplot(313)
    # ax1.scatter(background_points[:, 1], background_points[:, 0], c='lightgray', alpha=0.2)
    ax2.scatter(background_points[:, 1], background_points[:, 2], c='lightgray', alpha=0.2)
    # ax3.scatter(background_points[:, 0], background_points[:, 2], c='lightgray', alpha=0.2)

    MIN, MAX = coordinates.min(axis=0), coordinates.max(axis=0)
    blacks = coordinates[~mask]
    whites = coordinates[mask]
    ax2.fill([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
             [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='white')
    ax2.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
             [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='orange')

    # ax1.scatter(whites[:, 1], whites[:, 0], s=0.1, c='white')
    # ax2.scatter(whites[:, 1], whites[:, 2], s=0.01, c='white')
    # ax3.scatter(whites[:, 0], whites[:, 2], s=0.1, c='white')

    # ax1.scatter(blacks[:, 1], blacks[:, 0], s=0.1, c='k')
    ax2.scatter(blacks[:, 1], blacks[:, 2], s=0.005, c='k')
    # ax3.scatter(blacks[:, 0], blacks[:, 2], s=0.1, c='k')

    # ax1.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
    #          [MIN[0], MAX[0], MAX[0], MIN[0], MIN[0]], c='orange')

    ax2.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
             [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='orange')

    # ax3.plot([MIN[0], MIN[0], MAX[0], MAX[0], MIN[0]],
    #          [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='orange')

    # ax1.invert_xaxis()
    # ax2.set_title(indx)
    ax2.invert_xaxis()
    plt.tight_layout()

    plt.savefig(f'{folder_out}/{indx}.png', dpi=120)
    plt.close()


def plot_lidar_chessboard(
        chessboards: list,
        background_points: np.ndarray,
        indx: int,
        folder_out: str
):
    plt.figure(figsize=(10, 5))
    # ax1 = plt.subplot(311)
    ax2 = plt.subplot(111)
    # ax3 = plt.subplot(313)
    # ax1.scatter(background_points[:, 1], background_points[:, 0], c='lightgray', alpha=0.2)
    ax2.scatter(background_points[:, 1], background_points[:, 2], c='lightgray', alpha=0.2)
    # ax3.scatter(background_points[:, 0], background_points[:, 2], c='lightgray', alpha=0.2)

    for chessboard in chessboards:
        MIN, MAX = chessboard['cloud'].min(axis=0), chessboard['cloud'].max(axis=0)
        blacks = chessboard['cloud'][chessboard['intensity'] <= chessboard['median_intensity_threshold']]
        whites = chessboard['cloud'][chessboard['intensity'] > chessboard['median_intensity_threshold']]
        ax2.fill([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
                 [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='white')
        ax2.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
                 [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='orange')
        # ax1.scatter(blacks[:, 1], blacks[:, 0], s=1, c='r')
        ax2.scatter(blacks[:, 1], blacks[:, 2], s=0.05, c='black')
        # ax3.scatter(blacks[:, 0], blacks[:, 2], s=1, c='r')

        # ax1.scatter(whites[:, 1], whites[:, 0], s=1, c='dodgerblue')
        # ax2.scatter(whites[:, 1], whites[:, 2], s=0.05, c='white')
        # ax3.scatter(whites[:, 0], whites[:, 2], s=1, c='dodgerblue')

        # ax1.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
        #          [MIN[0], MAX[0], MAX[0], MIN[0], MIN[0]], c='orange')



        # ax3.plot([MIN[0], MIN[0], MAX[0], MAX[0], MIN[0]],
        #          [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]], c='orange')

    # ax1.invert_xaxis()
    # ax1.set_title(indx)
    ax2.invert_xaxis()
    plt.tight_layout()

    plt.savefig(f'{folder_out}/{indx}.png', dpi=120)
    plt.close()
