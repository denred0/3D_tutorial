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
    centroids = {}
    max_plane_idx = [20]  # количество плоскостей, которые будем искать
    plane_distance_threshold = [0.01, 0.02]
    dbscan_threshold = [0.01, 0.02]

    wcss_results = {}

    count_of_experimrnts = len(max_plane_idx) * len(plane_distance_threshold) * len(dbscan_threshold)
    current_experiment = 0

    # будем искать лучшие параметры для кластеризации
    for plane_idx in max_plane_idx:
        for dbscan_th in dbscan_threshold:
            for plane_distance_th in plane_distance_threshold:

                rest = pcd
                for i in tqdm(range(plane_idx)):
                    colors = plt.get_cmap("tab20")(i)
                    # с помощью RANSAC находим поддерживающие плоскости
                    segment_models[i], inliers = rest.segment_plane(
                        distance_threshold=plane_distance_th, ransac_n=3, num_iterations=1000)
                    # назначаем найденным сегментом все невыбросы, а найденную плоскость
                    segments[i] = rest.select_by_index(inliers)

                    # в найденной плоскости делаем кластеризацию т.к. эта плоскость может идти через всю картину
                    # и в одной плоскости могут лежать точки из разных частей 3D сцены
                    # используем алгоритм DBSCAN, которому передаем на каком расстоянии искать точки одного
                    # кластера и сколько соседних точек должно быть.
                    labels = np.array(segments[i].cluster_dbscan(eps=dbscan_th * 10, min_points=10))

                    # находим количество точек в каждом кластере
                    candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
                    # находим метку кластера с наибольшим количеством точек
                    best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

                    # в оставшиеся точки (которые не попали на плоскость) добавляем и точки нашей плоскости,
                    # которые не входят в самый большой кластер
                    rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
                        list(np.where(labels != best_candidate)[0]))

                    # итоговый сегмент это все точки самого большого кластера
                    segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))
                    # красим точки сегмента
                    segments[i].paint_uniform_color(list(colors[:3]))

                    # считаем центроид кластера, он нужен будет для суммы расчета квадратов расстояний
                    # центроида и всех точек кластера
                    centroids[i] = np.mean(np.asarray(segments[i].points), axis=0)

                # рассчитываем сумму квадратов расстояний от центроида каждого кластера до его точек
                wcss = 0
                for segment, centroid in zip(segments.values(), centroids.values()):
                    wcss += np.sum(np.linalg.norm(centroid - np.asarray(segment.points), axis=1))

                # сохраняем для постороения графика
                wcss_results[f"pc_{plane_idx}_dpth{plane_distance_th}_dbth_{dbscan_th}_"] = wcss

                # оставшиеся точки после поиска всех плоскостей кластеризируем с помощью DBSCAN
                labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
                max_label = labels.max()
                colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
                colors[labels < 0] = 0
                rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

                # o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest],
                #                                   width=1080,
                #                                   height=720,
                #                                   zoom=0.3199,
                #                                   front=[0.30159062875123849, 0.94077325609922868, 0.15488309545553303],
                #                                   lookat=[-3.9559999108314514, -0.055000066757202148, -0.27599999308586121],
                #                                   up=[-0.044411423633999815, -0.138726419067636, 0.98753122516983349])
                current_experiment += 1
                print(f"Experiment {current_experiment}/{count_of_experimrnts}")

    # строим график зависимости параметров от суммы квадратов расстояний кластеров
    plt.plot(list(wcss_results.keys()), list(wcss_results.values()))
    plt.xticks(rotation=90)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()


if __name__ == "__main__":
    # step_by_step()
    one_function_clustering()
