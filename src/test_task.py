import numpy as np
import open3d as o3d
from my_utils import voxel_sampling, random_sampling
import pickle
import json
from sklearn.cluster import DBSCAN


def voxel_clustering(points: np.ndarray,
                     colors: np.ndarray,
                     voxel_size: float,
                     type='barycenter') -> [np.array, np.array, np.array]:
    nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)
    # nb_vox.astype(int) #this gives you the number of voxels per axis

    # np.min(points, axis=0) - находим минимальное значение в каждом столбце (минимальное X, Y, Z)
    # points - np.min(points, axis=0) - вычитаем из  каждой координаты минимальное - таким образом приводим все координаты к началу координат (0, 0, 0)
    # ((points - np.min(points, axis=0)) // voxel_size).astype(int) - определяем к какой клеточке нашего воксельного разбиения относится каждая точка
    # return_inverse=True - берет индексы уникальных значений non_empty_voxel_keys в оригинальном массиве и составляет из них массив,
    # размер, которого равен оригинальному, показывая в каких позициях оригинального массива надо вставить эти уникальные значения
    # чтобы восстановить оригинальный массив
    # return_counts - возвращает количество для каждого уникального значения. В гашем случае показывает, сколько точек входит в один воксель.
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)

    # idx_pts_vox_sorted = np.argsort(inverse) возвращает индексы отсортированного массива inverse
    # inverse показывает индексы точек, входящие в каждый воксель из non_empty_voxel_keys
    # Например, у нас inverse равен [27162 35807 27162... это означает, что первая точка входит в воксель с индексом 27162,
    # вторая входит в воксель с индексом 35807 и т.д.
    # Когда мы вызываем np.argsort(inverse) мы мысленно сортируем inverse, так чтобы он был [0, 0, 0, 1, 1, 1..]
    # таким образом, чтобы все точки входящие в 0-ой воксель были в начале
    # но вместо значений inverse возвращаются индексы этих точек, входящих в 0-ой воксель.
    # На выходе мы получаем массив у которого в начале индексы точек, входящие в 0-ой воксель, потом в 1-ый.
    # Нам это необходимо, т.к. non_empty_voxel_keys возвращается в отсортированном виде и мы получив idx_pts_vox_sorted
    # сможем потом итерировать по вокселям и по точкам
    idx_pts_vox_sorted = np.argsort(inverse)

    voxel_grid = {}
    voxel_grid_colors = {}
    last_seen = 0
    center_points, cluster_points, cluster_colors = [], [], []
    # grid_candidate_center, grid_candidate_center_colors = {}, {}

    # non_empty_voxel_keys - это уникальные координаты вокселей, в !отсортированном виде в порядке возрастания
    # т.е. первый элемент в этом массиве - это первый воксель в нашей 3D сетке
    for idx, vox in enumerate(non_empty_voxel_keys):
        # берем все точки, индексы которых равны idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]
        # а это ни что иное как индексы точек, входящих в первый воксель. Мы берем их количеством равным nb_pts_per_voxel
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]
        voxel_grid_colors[tuple(vox)] = colors[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]

        if type == "barycenter":
            center_points.append(np.mean(voxel_grid[tuple(vox)], axis=0))
            cluster_points.append(voxel_grid[tuple(vox)])
            cluster_colors.append(voxel_grid_colors[tuple(vox)])

        elif type == "candidate_center":
            center_points.append(
                voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]
                                                      - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
            cluster_points.append(voxel_grid[tuple(vox)])
            cluster_colors.append(voxel_grid_colors[tuple(vox)])

        last_seen += nb_pts_per_voxel[idx]

    return center_points, cluster_points, cluster_colors


def dbscan_clustering(points, colors, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    print("Done")


def save_csvs(center_points, cluster_points, cluster_colors):
    np.savetxt("center_points.csv", center_points, delimiter=";")
    np.savetxt("cluster_points.csv", cluster_points, delimiter=";")
    np.savetxt("cluster_colors.csv", cluster_colors, delimiter=";")


if __name__ == "__main__":
    data_path = "data/test_task/fovea_tikal_guatemala_pcloud.asc"
    raw_data = np.loadtxt(data_path)

    type = "voxel"  # voxel
    radius = 0.5
    dbscan_min_samples = 10

    points = raw_data[:, :3]
    colors = raw_data[:, 3:6] / 255

    if type == "dbscan":
        center_points, cluster_points, cluster_colors = dbscan_clustering(points, colors, radius, dbscan_min_samples)
    elif type == "voxel":
        center_points, cluster_points, cluster_colors = voxel_clustering(points, colors, radius)
        save_csvs(center_points, cluster_points, cluster_colors)
    else:
        print(f"{type} clustering doesn't exist. Please choose 'voxel' or 'dbscan'")

    # print(f"Points before: {len(points)}")
    # # voxel sampling
    # voxel_size = 1.5
    # points, colors = voxel_sampling(points, colors, voxel_size=voxel_size, type="barycenter")
    # print(f"Points after: {len(points)}")
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    # pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    # o3d.visualization.draw_geometries([pcd])
