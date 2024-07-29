import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d

# Read 3D points from file
points = np.loadtxt('3d_points.txt')

# Create a point cloud from the 3D points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])

# print the size of pcd points
print("Size of pcd points: ", len(pcd.points))


# Visualize the point cloud with background color black and reduce the point size
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 1.2
vis.run()
vis.destroy_window()