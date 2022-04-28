# data https://drive.google.com/file/d/12Iy4fkJ1i1Xh-dzGvsf_M66e8eVa1vyx/view
# article https://medium.com/towards-data-science/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c
import numpy as np
import laspy as lp
import os
import matplotlib.pyplot as plt
import open3d as o3d

from my_utils import voxel_sampling, random_sampling

# from mpl_toolkits import mplot3d


if __name__ == "__main__":
    input_path = "data/lidar_point_cloud_processing/input"
    output_path = "data/lidar_point_cloud_processing/output"
    dataname = "NZ19_Wellington.las"
    point_cloud = lp.read(os.path.join(input_path, dataname))

    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

    # random sampling
    # sampled_points, sampled_colors = random_sampling(points, colors, factor=160)

    # voxel sampling
    voxel_size = 6
    sampled_points, sampled_colors = voxel_sampling(points, colors, voxel_size=voxel_size, type="barycenter")

    print(f"Points count before sampling: {len(points)}")
    print(f"Points count after sampling: {len(sampled_points)}")

    sampled_points_colors = np.hstack((np.asarray(sampled_points), np.asarray(sampled_colors)))

    # save result
    np.savetxt(os.path.join(output_path, dataname + "_voxel-best_point_%s.xyz" % (voxel_size)), sampled_points_colors,
               delimiter=";",
               fmt="%s")

    # plot result
    ax = plt.axes(projection='3d')
    ax.scatter(np.asarray(sampled_points)[:, 0],
               np.asarray(sampled_points)[:, 1],
               np.asarray(sampled_points)[:, 2],
               c=np.asarray(sampled_colors) / np.max(sampled_colors),
               s=0.01)
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points_colors[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(sampled_points_colors[:, 3:6] / np.max(sampled_colors))
    o3d.visualization.draw_geometries([pcd],
                                      width=1080,
                                      height=720)
