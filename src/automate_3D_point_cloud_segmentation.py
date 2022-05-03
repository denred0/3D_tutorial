# https://towardsdatascience.com/how-to-automate-3d-point-cloud-segmentation-and-clustering-with-python-343c9039e4f5
# https://drive.google.com/file/d/1CJrH9eqzOte3PHJ_g8iLifJFE_cBGg6b/view

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm


def step_by_step():
    pcd = o3d.io.read_point_cloud("data/automate_3D_point_cloud_segmentation/input/TLS_kitchen.ply")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16),
                         fast_normal_computation=True)

    # RANSAC - алгоритм для поиска выбросов и не выбросов. Задается количество точек ransac_n для задания самой поддерживаемой
    # плоскости (плоскости, на которую попадает максимальное количество точек с заданной точностью distance_threshold.
    # Выполняется num_iterations для поиска этой плоскости. Возвращаются a, b, c точки для создания плоскости и
    # индексы всех точек, которые на нее попали.
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([outlier_cloud, inlier_cloud],
                                      width=1080,
                                      height=720)

    pcd = o3d.io.read_point_cloud("data/automate_3D_point_cloud_segmentation/input/TLS_kitchen_sample.ply")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16),
                         fast_normal_computation=True)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([pcd],
                                      width=1080,
                                      height=720)

    # DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))  # 5 сантиметров
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


def one_function_clustering():
    pcd = o3d.io.read_point_cloud("data/automate_3D_point_cloud_segmentation/input/TLS_kitchen.ply")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16),
                         fast_normal_computation=True)

    segment_models = {}  # сохраняем сюда параметры a, b, c найденных плоскостей
    segments = {}
    max_plane_idx = 20  # количество плоскостей, которые будем искать
    d_threshold = 0.01

    rest = pcd
    for i in tqdm(range(max_plane_idx)):
        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        segments[i] = rest.select_by_index(inliers)
        labels = np.array(segments[i].cluster_dbscan(eps=d_threshold * 10, min_points=10))
        candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
        best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

        rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
            list(np.where(labels != best_candidate)[0]))
        segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))

        segments[i].paint_uniform_color(list(colors[:3]))
        # rest = rest.select_by_index(inliers, invert=True)

    labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest],
                                      width=1080,
                                      height=720,
                                      zoom=0.3199,
                                      front=[0.30159062875123849, 0.94077325609922868, 0.15488309545553303],
                                      lookat=[-3.9559999108314514, -0.055000066757202148, -0.27599999308586121],
                                      up=[-0.044411423633999815, -0.138726419067636, 0.98753122516983349])


if __name__ == "__main__":
    # step_by_step()
    one_function_clustering()
