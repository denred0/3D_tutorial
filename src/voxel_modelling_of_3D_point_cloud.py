# https://medium.com/towards-data-science/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227

import numpy as np
import open3d as o3d
import laspy as lp
import os
from tqdm import tqdm

input_path = "data/voxel_modelling_of_3D_point_cloud/input"
output_path = "data/voxel_modelling_of_3D_point_cloud/output"
dataname = "2021_heerlen_table.las"

# читаем облако точек. las формат laspy
point_cloud = lp.read(os.path.join(input_path, dataname))

# перегоняем из las в формат open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose())
pcd.colors = o3d.utility.Vector3dVector(
    np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose() / 65535)

# переводим в воксели, размер завязан на размер ограничивающей рамки облака точек
voxel_size = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.01, 4)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# берем только воксели со значениями
voxels = voxel_grid.get_voxels()

# переведем из вокселей в меш, чтобы можно было выгрузить и загрузить в программы, работающие с мешами
vox_mesh = o3d.geometry.TriangleMesh()

# бежим по всем не пустым вокселям, создаем кубик (1, 1, 1), красим его в цвет вокселя и перемещаем на то место,
# какие координаты у вокселя
for v in tqdm(voxels):
    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube.paint_uniform_color(v.color)
    cube.translate(v.grid_index, relative=False)
    vox_mesh += cube

# когда создавали voxel_grid, то координатой вокселя был левый нижний угол. Передвинем на середину вокселя
vox_mesh.translate([0.5, 0.5, 0.5], relative=True)

# изменим размер, чтобы привести к размеру исходного облака точек
vox_mesh.scale(voxel_size, [0, 0, 0])

#передвинем в начало координат, хотя у нас и так воксели вроде на нужных местах уже стоят.
vox_mesh.translate(voxel_grid.origin, relative=True)

# объединим пересекающиеся вершины
vox_mesh.merge_close_vertices(0.0000001)

# сохраним
o3d.io.write_triangle_mesh(os.path.join(output_path, "voxel_mesh_h.ply"), vox_mesh)

o3d.visualization.draw_geometries([vox_mesh],
                                  width=1080,
                                  height=720)
