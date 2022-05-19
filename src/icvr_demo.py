import open3d as o3d
import numpy as np

# with open('data/icvr/input/Scan2.txt') as f:
#     lines = f.readlines()
#
# print(f"Points: {lines[0]}")
#
# coords = lines[1].split(",")
# x = np.asarray(coords[0::3][:-1])
# y = np.asarray(coords[1::3])
# z = np.asarray(coords[2::3])
#
# points = np.vstack((x, y, z)).transpose()
#
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

pcd = o3d.io.read_point_cloud("target.pcd")
o3d.visualization.draw_geometries([pcd],
                                  width=1080,
                                  height=720)

print()
