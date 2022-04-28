import numpy as np


def random_sampling(points, colors, factor):
    # берем каждую точку через factor раз
    decimated_points_random = points[::factor]
    decimated_colors_random = colors[::factor]

    return decimated_points_random, decimated_colors_random


def voxel_sampling(points, colors, voxel_size, type):
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
    grid_barycenter, grid_candidate_center = [], []
    grid_barycenter_colors, grid_candidate_center_colors = [], []
    last_seen = 0

    # non_empty_voxel_keys - это уникальные координаты вокселей, в !отсортированном виде в порядке возрастания
    # т.е. первый элемент в этом массиве - это первый воксель в нашей 3D сетке
    for idx, vox in enumerate(non_empty_voxel_keys):
        # берем все точки, индексы которых равны idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]
        # а это ни что иное как индексы точек, входящих в первый воксель. Мы берем их количеством равным nb_pts_per_voxel
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]
        voxel_grid_colors[tuple(vox)] = colors[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]

        if type == "barycenter":
            # берем среднее по всем точкам
            grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
            grid_barycenter_colors.append(np.mean(voxel_grid_colors[tuple(vox)], axis=0))
        elif type == "candidate_center":
            # берем точку, ближайшую к центру всех точек
            # вычитаем из каждой координаты среднее np.mean(voxel_grid[tuple(vox)], axis=0) axis=0 - среднее по столбцам
            # np.linalg.norm( ... axis=1) - находит норму (длину вектора) для каждой строки
            # argmin() - находит индекс вектора с минимальной длинной. Это и есть индекс ближайшей точки
            grid_candidate_center.append(voxel_grid[tuple(vox)][
                                             np.linalg.norm(
                                                 voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0),
                                                 axis=1).argmin()])
            grid_candidate_center_colors.append(voxel_grid_colors[tuple(vox)][
                                                    np.linalg.norm(
                                                        voxel_grid_colors[tuple(vox)] - np.mean(
                                                            voxel_grid_colors[tuple(vox)],
                                                            axis=0),
                                                        axis=1).argmin()])
        last_seen += nb_pts_per_voxel[idx]

    if type == "barycenter":
        return grid_barycenter, grid_barycenter_colors
    elif type == "candidate_center":
        return grid_candidate_center, grid_candidate_center_colors
