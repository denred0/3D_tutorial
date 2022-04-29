# https://drive.google.com/file/d/1DUEy-gayYudkuJUou_cfNmPAmwdbbgml/view
# https://towardsdatascience.com/guide-to-real-time-visualisation-of-massive-3d-point-clouds-in-python-ea6f00241ee0

import numpy as np
import laspy as lp
import os
import pptk
import open3d as o3d

from my_utils import voxel_sampling


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
    v = pptk.viewer(points)
    v.attributes(colors / 65535)
    v.set(point_size=0.001, bg_color=[0, 0, 0, 0], show_axis=0, show_grid=0)


def open3d_visualize(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 65535)
    # pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.40)
    o3d.visualization.draw_geometries([voxel_grid])


if __name__ == "__main__":
    input_path = "data/visualize_point_cloud_tutorial/input"
    dataname = "2020_Drone_M"
    point_cloud = lp.read(os.path.join(input_path, dataname + ".las"))

    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

    # voxel sampling
    voxel_size = 0.1
    sampled_points, sampled_colors = voxel_sampling(points, colors, voxel_size=voxel_size, type="barycenter")

    # pptk_visualize(np.asarray(sampled_points), np.asarray(sampled_colors))

    open3d_visualize(np.asarray(sampled_points), np.asarray(sampled_colors))
