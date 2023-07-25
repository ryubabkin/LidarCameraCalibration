import matplotlib.pyplot as plt
from tools.utils import *
import scipy.cluster.hierarchy as hcluster

# %%

folder = '/Users/brom/Laboratory/GlobalLogic/MEAA/DATA/03_02_rover_data_orange'
image_folder = f'{folder}/camera_normal/'
lidar_folder = f'{folder}/lidar_normal/'

# id_1 = '001600'
min_delta = 0.1
max_distance = 15
median_distance_stop = 0.004
cluster_threshold = 0.5
min_points_in_cluster = 20

# id_2 = '001865'

# image_1, pcd_1, intensity_1 = read_input_pair(image_folder, lidar_folder, id_1)
# points_1 = np.asarray(pcd_1.points)
# intensities_1 = pcd_1.intensities.numpy()
# ids_train = sorted(os.listdir(image_folder))[330:370]
# ids = sorted(os.listdir(image_folder))[380:480]

# ids_train = sorted(os.listdir(image_folder))[:190]
ids = sorted(os.listdir(image_folder))[:480]

stop_id = 370

background_pcd = o3d.geometry.PointCloud()
points_list = None
medians = []
means = []
for indx, id in enumerate(ids[:stop_id]):
    image, pcd, intensity = read_input_pair(image_folder, lidar_folder, id.split('.')[0])

    points = np.asarray(pcd.points)
    dist = np.sqrt(np.sum(points ** 2, axis=1))
    points = points[dist < max_distance]

    # pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if points_list is None:
        points_list = points
        median = 100
    else:
        distances = np.asarray(pcd.compute_point_cloud_distance(background_pcd))
        median = np.median(distances)
        medians.append(np.median(distances))
        means.append(np.mean(distances))
        points_list = np.concatenate([points_list, points], axis=0)
    background_pcd.points = o3d.utility.Vector3dVector(points_list)
    background_pcd = background_pcd.remove_duplicated_points()
    background_points = np.asarray(background_pcd.points)
    if median < median_distance_stop:
        print('==========')
        print(f'index {indx}, median = {np.round(median, 5)}')
        print('==========')
        break

plt.figure()
plt.plot(medians)
plt.plot(means)
plt.show()
background_points = np.asarray(background_pcd.points)
for indx, id in enumerate(ids[stop_id:]):
    image, pcd, intensity = read_input_pair(image_folder, lidar_folder, id.split('.')[0])
    points = np.asarray(pcd.points)
    dist = np.sqrt(np.sum(points ** 2, axis=1))
    points = points[dist < max_distance]

    # pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    distances = np.asarray(pcd.compute_point_cloud_distance(background_pcd))
    difference = points[distances > min_delta]

    clusters = hcluster.fclusterdata(
        X=difference,
        t=cluster_threshold,
        criterion="distance"
    )
    unique, counts = np.unique(clusters, return_counts=True)
    obstacle_clouds = [difference[clusters == c] for (c, n) in zip(unique, counts) if n >= min_points_in_cluster]

    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.scatter(background_points[:, 1], background_points[:, 0], c='lightgray', alpha=0.2)
    ax2.scatter(background_points[:, 1], background_points[:, 2], c='lightgray', alpha=0.2)
    for obstacle in obstacle_clouds:
        Q75 = np.quantile(obstacle, 0.75, axis=0)
        Q25 = np.quantile(obstacle, 0.25, axis=0)
        IQR = Q75 - Q25
        obstacle = obstacle[
            (obstacle[:, 0] < Q75[0] + 1.5 * IQR[0]) &
            (obstacle[:, 0] > Q25[0] - 1.5 * IQR[0]) &
            (obstacle[:, 1] < Q75[1] + 1.5 * IQR[1]) &
            (obstacle[:, 1] > Q25[1] - 1.5 * IQR[1]) &
            (obstacle[:, 2] < Q75[2] + 1.5 * IQR[2]) &
            (obstacle[:, 2] > Q25[2] - 1.5 * IQR[2])
            ]
        MIN = obstacle.min(axis=0)
        MAX = obstacle.max(axis=0)
        ax1.scatter(obstacle[:, 1], obstacle[:, 0])
        ax1.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
                 [MIN[0], MAX[0], MAX[0], MIN[0], MIN[0]])

        ax2.scatter(obstacle[:, 1], obstacle[:, 2])
        ax2.plot([MIN[1], MIN[1], MAX[1], MAX[1], MIN[1]],
                 [MIN[2], MAX[2], MAX[2], MIN[2], MIN[2]])
    ax1.set_ylim(3.5, 14)
    ax1.set_xlim(14, -14)
    ax1.set_title(stop_id + indx)

    ax2.set_ylim(-2, 4)
    ax2.set_xlim(14, -14)
    ax2.axis("equal")
    plt.tight_layout()
    plt.savefig(f'/Users/brom/Laboratory/GlobalLogic/MEAA/DATA/03_02_rover_data_orange/pictures_6/{id}')
    # plt.show()
    plt.close()

#
# # visualize([target_pcd])
# for id in ids:
#     image_2, pcd_2, intensity_2 = read_input_pair(image_folder, lidar_folder, id.split('.')[0])
#     points_2 = np.asarray(pcd_2.points)
#     # intensities_2 = pcd_2.intensities.numpy()
#
#     # pcd_1.paint_uniform_color([0,0,1])
#     # pcd_2.paint_uniform_color([0.5,0.5,0])
#     # visualize([pcd_1, pcd_2])
#
#     # dist_pc1_pc2 = np.asarray(pcd_1.compute_point_cloud_distance(pcd_2))
#     dist_pc2_pc1 = np.asarray(pcd_2.compute_point_cloud_distance(background_pcd))
#
#     # df12 = pd.DataFrame({"distances": dist_pc1_pc2}) # transform to a dataframe
#     df21 = pd.DataFrame({"distances": dist_pc2_pc1})  # transform to a dataframe
#     # plt.figure()
#     # ax = plt.subplot(111)
#     # df12.plot(kind="hist", alpha=0.5, bins = 100, log=True, ax=ax, color='b') # HISTOGRAM
#     # df21.plot(kind="hist", alpha=0.5, bins = 100, log=True, ax=ax, color='r') # HISTOGRAM
#     #
#     # plt.show()
#
#     plt.figure(figsize=(10, 15))
#     ax1 = plt.subplot(211)
#     ax2 = plt.subplot(212)
#     ax1.scatter(background_points[:, 1], background_points[:, 0], c='lightgray', alpha=0.2)
#     ax1.scatter(points_2[:, 1], points_2[:, 0])
#     ax1.scatter(points_2[:, 1][dist_pc2_pc1 > min_delta], points_2[:, 0][dist_pc2_pc1 > min_delta], c='r')
#     ax1.set_ylim(3.5, 14)
#     ax1.set_xlim(0, -5)
#     # ax1.axis("off")
#     ax2.scatter(background_points[:, 1], background_points[:, 2], c='lightgray', alpha=0.2)
#     ax2.scatter(points_2[:, 1], points_2[:, 2])
#     ax2.scatter(points_2[:, 1][dist_pc2_pc1 > min_delta], points_2[:, 2][dist_pc2_pc1 > min_delta], c='r')
#     ax2.set_ylim(-2, 4)
#     ax2.set_xlim(0, -5)
#     # ax2.axis("off")
#     plt.tight_layout()
#     plt.savefig(f'/Users/brom/Laboratory/GlobalLogic/MEAA/DATA/03_02_rover_data_orange/pictures_3/{id}')
#     # plt.show()
#     plt.close()
