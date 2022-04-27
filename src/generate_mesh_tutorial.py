import numpy as np
import open3d as o3d
import os


def lod_mesh_export(mesh, lods, extension, path, algo):
    # mesh extension: .ply, .obj, .stl or .gltf
    # Levels of Details (LoD) - количество треугольников
    mesh_lods = {}
    for i in lods:
        # уменьшаем количество треугольников до количества i
        mesh_lod = mesh.simplify_quadric_decimation(i)

        # удаляем артефакты
        mesh_lod.remove_degenerate_triangles()
        mesh_lod.remove_duplicated_triangles()
        mesh_lod.remove_duplicated_vertices()
        mesh_lod.remove_non_manifold_edges()

        o3d.io.write_triangle_mesh(os.path.join(path, f"{algo}_lod_{i}.{extension}"), mesh_lod)
        mesh_lods[i] = mesh_lod
        print(f"Generation of {i} Levels of Details (LoD) successful")
    return mesh_lods


def ball_pivoting_algorithm_mesh(pcd, output_path):
    # Ball-Pivoting Algorithm
    # Катаем шар между точками облака. Радиус шара чуть больше расстояния между точками облака.
    # Когда шар попадает на 3 точки, мы формируем треугольник
    # Необходимо вычислить радиус шара. Для этого считаем среднее расстояние между точками.

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist  # ball radius
    print(f"Avg Ball-Pivoting Algorithm ball radius: {radius}")
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))

    my_lods = lod_mesh_export(bpa_mesh, [100000, 50000, 10000, 1000, 100], "ply", output_path, "bpa")

    o3d.visualization.draw_geometries([my_lods[100]])


def poisson_mesh(pcd, output_path):
    # Реконструкция Пуассона.
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                             depth=8,
                                                                             width=0,
                                                                             scale=1.1,
                                                                             linear_fit=False)[0]
    # получаем bbox, описывающий изначальное облако точек
    bbox = pcd.get_axis_aligned_bounding_box()
    # фильтруем mesh чтобы оставить только исходный объект
    p_mesh_crop = poisson_mesh.crop(bbox)

    my_lods = lod_mesh_export(p_mesh_crop, [100000], "ply", output_path, "poisson")

    o3d.visualization.draw_geometries([my_lods[100000]])


if __name__ == "__main__":
    input_path = "data/generate_mesh_tutorial/input"
    output_path = "data/generate_mesh_tutorial/output"
    dataname = "sample_w_normals.xyz"
    point_cloud = np.loadtxt(os.path.join(input_path, dataname), skiprows=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 6:9])
    # o3d.visualization.draw_geometries([pcd], width=1024, height=720)

    # Ball-Pivoting Algorithm
    # ball_pivoting_algorithm_mesh(pcd, output_path)

    # Poisson’ reconstruction
    poisson_mesh(pcd, output_path)
