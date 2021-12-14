
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pdb
import torch
import trimesh

from sdf import *
from string import ascii_uppercase
from plb.algorithms.sample_data import *
from data_utils import load_data

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
    ax.view_init(45, 135)
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c='b', s=20)
    ax.scatter(shapes[:, 0], shapes[:, 2], shapes[:, 1], c='r', s=20)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.invert_yaxis()
    plt.savefig(path)
    # plt.show()

if __name__ == "__main__":
    # FONT = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'

    # w, h = measure_image(IMAGE)

    # f = rounded_box((w * 1.1, h * 1.1, 0.1), 0.05)
    # f |= image(IMAGE).extrude(1) & slab(z0=0, z1=0.075)

    update = True
    debug = False
    prefix = 'shapes/alphabet'
    if prefix == 'shapes/simple':
        image_names = ['fish', 'clover', 'heart', 'flower', 'moon', 'controller', 'hat', 'nut', 'butterfly']
    elif prefix == 'shapes/alphabet':
        image_names = list(ascii_uppercase)
    else:
        raise NotImplementedError
    shape_size = (0.25, 0.15, 0.25)
    shape_pos = (0.5, 0.125, 0.5)
    # dataset_image_path = f'shapes/alphabet_dataset.png'
    # dataset_image = cv2.imread(dataset_image_path)
    for i, n in enumerate(image_names):
        # print(n)
        if debug and i > 0: break

        point_cloud_path = f'{prefix}/{n}/{n}.ply'
        if not os.path.exists(point_cloud_path) or update:
            # pdb.set_trace()
            image_path = f'{prefix}/{n}/{n}.png'
            # cv2.imwrite(image_path, dataset_image[int((i // 9) * size) : int((i // 9) * size + size), int((i % 9) * size) : int((i % 9) * size + size)])
            scaled_image_path = f'{prefix}/{n}/{n}_scaled.png'
            
            orig_image = cv2.imread(image_path)
            size = min(orig_image.shape[0], orig_image.shape[1])
            orig_image = cv2.resize(orig_image, (size, size))
            cv2.imwrite(scaled_image_path, orig_image)

            scaled_image = cv2.imread(scaled_image_path)
            gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
            pixels = cv2.countNonZero(thresh)
            image_area = scaled_image.shape[0] * scaled_image.shape[1]
            size_ratio = np.sqrt(pixels / image_area)

            w, h = measure_image(scaled_image_path)
            f = image(scaled_image_path).scale((shape_size[0] / size_ratio / w, shape_size[2] / size_ratio / h))\
                                        .extrude(shape_size[1]).orient(Y).translate(shape_pos)
            f.save(point_cloud_path, step=0.01)

        h5_path = f'{prefix}/{n}/{n}.h5'

        if not os.path.exists(h5_path) or update:
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            upper = pcd.get_max_bound()
            lower = pcd.get_min_bound()

            tri_mesh = trimesh.load_mesh(point_cloud_path)

            sampled_points = sampling(lower, upper, tri_mesh, [], [], n_particle_sample)

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
