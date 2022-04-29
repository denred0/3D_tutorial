# https://drive.google.com/file/d/1DUEy-gayYudkuJUou_cfNmPAmwdbbgml/view
# https://towardsdatascience.com/guide-to-real-time-visualisation-of-massive-3d-point-clouds-in-python-ea6f00241ee0

import numpy as np
import laspy as lp
import os
import pptk
import open3d as o3d

from my_utils import voxel_sampling, random_sampling


# ------Google Colab Visualization----------

# ### PyntCloud ###
# conda install pyntcloud -c conda-forge
# from pyntcloud import PyntCloud
# pointcloud = PyntCloud.from_file("example.ply")
# pointcloud.plot()
# ### PyntCloud ###
# pip install pypotree
# import pypotree
# import numpy as np
# xyz = np.random.random((100000,3))
# cloudpath = pypotree.generate_cloud_for_display(xyz)
# pypotree.display_cloud_colab(cloudpath)

# --------Google Colab Visualization--------


def pptk_visualize(points, colors):
    # v = pptk.viewer(points)
    # v.attributes(colors / 65535)

    # находим нормали к каждой точке по 6 соседям. Определяется, как эти 6 соседей расположены в плосклости
    # значение нормали каждой точки это 3 значения, каждое из которых это наклон этой точки
    # к осям X, Y и Z соответственно. Если нормаль точки идет вдоль оси (колинеарно), то значение наклона стремится к 1
    # и наоборот, если нормаль перпендикулярна оси, то значение стремится к 0.
    normals = pptk.estimate_normals(points, k=6, r=np.inf)

    # оставляем только те индексы нормалей, где значение нормали для Z оси меньше 0.9,
    # т.е. нормаль не колинеарна оси Z, а нормали колинеарны оси Z только для плоских поверхностей,
    # у земли например.
    idx_normals = np.where(abs(normals[..., 2]) < 0.9)

    # оставив только не колинеарные к оси Z нормали мы убили и точки крыши авто или зданий т.к.
    # их нормали тоже колинеарны оси Z. Чтобы это исправить достанем все точки, значение Z которых
    # больше определенного значения - не земля
    idx_no_ground = np.where(points[..., 2] > np.min(points[..., 2] + 3.0))

    # найдем разницу между "не землей" и точками, нормали которых не колинеарны Z
    # это будут как раз точки крыши машины, потому что в нормалях все кроме земли и крыш авто, зданий,
    # а в "не земле" все кроме земли. Получается, у них разница только в точках крыш авто
    idx_wronglyfiltered = np.setdiff1d(idx_no_ground, idx_normals)

    # объединим нормали и точки крыш авто, зданий
    idx_retained = np.append(idx_normals, idx_wronglyfiltered)

    v = pptk.viewer(points[idx_retained], colors[idx_retained] / 65535)
    v.set(point_size=0.002, bg_color=[0, 0, 0, 0], show_axis=0, show_grid=0)


def open3d_visualize(points, colors, type, voxel_size=0.4):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 65535)
    if type == "points":
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd])
    elif type == "voxel":
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        o3d.visualization.draw_geometries([voxel_grid])


if __name__ == "__main__":
    input_path = "data/visualize_point_cloud_tutorial/input"
    dataname = "2020_Drone_M"
    point_cloud = lp.read(os.path.join(input_path, dataname + ".las"))

    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

    points, colors = random_sampling(points, colors, factor=5)

    # voxel sampling
    # voxel_size = 0.2
    # points, colors = voxel_sampling(points, colors, voxel_size=voxel_size, type="barycenter")

    pptk_visualize(np.asarray(points), np.asarray(colors))

    # open3d_visualize(np.asarray(sampled_points), np.asarray(sampled_colors), type="voxel", voxel_size=voxel_size)
