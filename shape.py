from sdf import *
from string import ascii_uppercase
from plb.algorithms.sample_data import *
import trimesh

n_particles = 300


if __name__ == "__main__":
    FONT = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    
    for c in ascii_uppercase:
        w, h = measure_text(FONT, c)
        f = text(FONT, c).scale(0.5).extrude(0.2).translate((0.5, 0.1, 0.5))
        shape_name = f'shapes/alphabet/{c}.ply'
        f.save(shape_name, step=0.01)

        tri_mesh = trimesh.load_mesh(shape_name)

        sampled_points = sampling(lower, upper, tri_mesh, [], n_particles)

        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

        # o3d.visualization.draw_geometries([sampled_pcd])

        cl, inlier_ind = sampled_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        sampled_pcd_t = sampled_pcd.select_by_index(inlier_ind)

        # o3d.visualization.draw_geometries([sampled_pcd_t])

        # cl, inlier_ind = sampled_pcd_t.remove_radius_outlier(nb_points=10, radius=0.02)
        # sampled_pcd_n = sampled_pcd_t.select_by_index(inlier_ind)

        # o3d.visualization.draw_geometries([sampled_pcd_n])

        sampled_points = np.asarray(sampled_pcd_t.points)

        # voxels = mesh_to_sdf.mesh_to_voxels(tri_mesh, 64, pad=True)
        # print(voxels)

        # vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        # mesh.show()

        # points, sdf = mesh_to_sdf.sample_sdf_near_surface(tri_mesh, surface_point_method='sample', number_of_points=250000)
        # print(sdf)

        # sampled_points = points[sdf < 0, :]

        # surface_pcd = mesh_to_sdf.get_surface_point_cloud(tri_mesh)

        # # colors = np.zeros(points.shape)
        # # colors[sdf < 0, 2] = 1
        # # colors[sdf > 0, 0] = 1
        # colors = np.zeros(sampled_points.shape)
        # colors[:, 2] = 1
        # cloud = pyrender.Mesh.from_points(sampled_points, colors=colors)
        # py_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        # # surface_mesh = pyrender.Mesh.from_points(surface_pcd.points)
        # scene = pyrender.Scene()
        # scene.add(cloud)
        # scene.add(py_mesh)
        # # scene.add(surface_mesh)
        # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

        sampled_points = fps(sampled_points, K=k_fps_particles, partial=False)
        # sampled_points.paint_uniform_color([0,0,0])
        # import pdb; pdb.set_trace()
        
        # sampled_pcd_fps = o3d.geometry.PointCloud()
        # sampled_pcd_fps.points = o3d.utility.Vector3dVector(sampled_points)
        # o3d.visualization.draw_geometries([sampled_pcd_fps])