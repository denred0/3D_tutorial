import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# pcd = o3d.io.read_point_cloud("target.pcd")


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud],
                                      width=1080,
                                      height=720)


# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
# display_inlier_outlier(pcd, ind)

with open('data/icvr/input/Scan4.txt') as f:
    lines = f.readlines()

print(f"Points: {lines[0]}")

coords = lines[1].split(",")
x = np.asarray(coords[0::3][:-1])
y = np.asarray(coords[1::3])
z = np.asarray(coords[2::3])

points = np.vstack((x, y, z)).transpose()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=45, std_ratio=0.2)
print(f"Points count: {np.asarray(pcd.select_by_index(ind).points).size}")
display_inlier_outlier(pcd, ind)



# X = np.asarray(pcd.points)
# kmeans = KMeans(n_clusters=2).fit(X)
# plt.scatter(x, y, c=kmeans.labels_, s=0.1)
# plt.show()


# lines = []
# for i in range(1, 28):
#     with open(f'data/icvr/prepared_data/vert{i}.txt') as f:
#         lines.append(f.readlines())
#
# coords = []
# for line in lines:
#     coords.extend(line[0].split(","))
# x = coords[::3]
# y = coords[1::3]
# z = coords[2::3]
#
# points = np.vstack((x, y, z)).transpose()
#
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
#
# o3d.io.write_point_cloud("target.pcd", pcd)


# o3d.visualization.draw_geometries([pcd],
#                                   width=1080,
#                                   height=720)
