from sdf import *
from string import ascii_uppercase
from plb.algorithms.sample_data import *
from data_utils import load_data

import matplotlib.pyplot as plt
import open3d as o3d
import torch
import trimesh

n_particle_sample = 2000
n_particle = 300

floor_pos = np.array(
    [[0.25, 0., 0.25], [0.25, 0., 0.5], [0.25, 0., 0.75],
    [0.5, 0., 0.25], [0.5, 0., 0.5], [0.5, 0., 0.75],
    [0.75, 0., 0.25], [0.75, 0., 0.5], [0.75, 0., 0.75]]
)

data_names = ['positions', 'shape_quats', 'scene_params']

def visualize_points(all_points, n_particles, path):
    # print(all_points.shape)
    points = all_points[:n_particles]
    shapes = all_points[n_particles:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c='b', s=20)
    ax.scatter(shapes[:, 0], shapes[:, 2], shapes[:, 1], c='r', s=20)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    plt.savefig(path)
    # plt.show()

if __name__ == "__main__":
    # FONT = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'

    # w, h = measure_image(IMAGE)

    # f = rounded_box((w * 1.1, h * 1.1, 0.1), 0.05)
    # f |= image(IMAGE).extrude(1) & slab(z0=0, z1=0.075)

    update = True
    debug = False
    prefix = 'shapes'
    image_names = ['butterfly', 'car', 'elephant', 'fish', 'flower', 'heart', 'house', 'panda', 'star']
    for i, n in enumerate(image_names):
        if debug and i > 0: break

        point_cloud_path = f'{prefix}/{n}/{n}.ply'
        if not os.path.exists(point_cloud_path) or update:
            image_path = f'{prefix}/{n}/{n}.png'
            w, h = measure_image(image_path)
            f = image(image_path).scale((0.33/w, 0.33/h)).extrude(0.175).orient(Y).translate((0.5, 0.1, 0.5))
            f.save(point_cloud_path, step=0.01)

        h5_path = f'{prefix}/{n}/{n}.h5'

        if not os.path.exists(h5_path) or update:
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            upper = pcd.get_max_bound()
            lower = pcd.get_min_bound()

            tri_mesh = trimesh.load_mesh(point_cloud_path)

            sampled_points = sampling(lower, upper, tri_mesh, [], n_particle_sample)

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

            sampled_points = fps(sampled_points, K=n_particle, partial=False)
            # sampled_points.paint_uniform_color([0,0,0])
            
            # sampled_pcd_fps = o3d.geometry.PointCloud()
            # sampled_pcd_fps.points = o3d.utility.Vector3dVector(sampled_points)
            # o3d.visualization.draw_geometries([sampled_pcd_fps])

            positions = np.concatenate([sampled_points, floor_pos])
            shape_quats = np.zeros((1, 4), dtype=np.float32)
            data = [positions, shape_quats, scene_params]

            store_data(data_names, data, h5_path)

        # import pdb; pdb.set_trace()
        goal_data = load_data(data_names, h5_path)
        goal_shape = torch.FloatTensor(goal_data[0]).unsqueeze(0)[:, :n_particle, :]
        visualize_points(goal_data[0], n_particle, f'{prefix}/{n}/{n}_sampled')
