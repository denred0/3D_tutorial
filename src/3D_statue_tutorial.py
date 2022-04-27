import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

file_data_path = "data/3d_statue_tutorial/sample.xyz"
point_cloud = np.loadtxt(file_data_path, skiprows=1, max_rows=1000000)
mean_Z = np.mean(point_cloud, axis=0)[2]
spatial_query = point_cloud[abs(point_cloud[:, 2] - mean_Z) < 1]
xyz = point_cloud[:, :3]
rgb = point_cloud[:, 3:]

ax = plt.axes(projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255, s=0.01)
plt.show()
