import copy
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import open3d as o3d
import pdb
import pymeshfix
import pyvista as pv
import rosbag
import ros_numpy
import scipy
import sys
import torch
import trimesh
import yaml

from datetime import datetime
from pysdf import SDF
from sensor_msgs.msg import PointCloud2
from timeit import default_timer as timer
from transforms3d.euler import euler2mat
from transforms3d.quaternions import *


mid_point = np.array([0.437, 0.0, 0.0])
floor_size = 0.05

bounding_r = 0.08 / 2 + 0.03 + 0.01

voxel_size = 0.01
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5


def eval_plot_curves(time_list, loss_list, path=''):
    loss_chamfer, loss_emd = loss_list
    plt.figure(figsize=[16, 9])
    plt.plot(time_list, loss_emd, linewidth=6, label='EMD')
    plt.plot(time_list, loss_chamfer, linewidth=6, label='Chamfer')
    plt.xlabel('time', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Control Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()


def o3d_visualize(display_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in display_list:
        vis.add_geometry(geo)
        vis.update_geometry(geo)
    
    # vis.get_render_option().light_on = False
    vis.get_render_option().point_size = 5
    vis.get_render_option().mesh_show_back_face = True
    # vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if o3d_write:
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        image_path = os.path.join(cd, '..', '..', 'images', f'{time_now}.png')
        
        vis.capture_screen_image(image_path)
        vis.destroy_window()
    else:
        vis.run()
        # o3d.visualization.draw_geometries(display_list, mesh_show_back_face=True)


def visualize_points(ax, all_points, n_points):
    points = ax.scatter(all_points[:n_points, 0], all_points[:n_points, 1], all_points[:n_points, 2], c='b', s=10)
    shapes = ax.scatter(all_points[n_points:, 0], all_points[n_points:, 1], all_points[n_points:, 2], c='r', s=10)
    
    ax.invert_yaxis()

    # centers = mid_point
    r = 0.075
    ax.set_xlim(mid_point[0] - r, mid_point[0] + r)
    ax.set_ylim(mid_point[1] - r, mid_point[1] + r)
    ax.set_zlim(mid_point[2] - r, mid_point[2] + r)

    # for ctr, dim in zip(centers, 'xyz'):
    #     getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    return points, shapes


def plt_render(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = len(particles_set)
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, rows * 3))
    row_titles = ['GT', 'Sample', 'Prediction']
    row_titles = row_titles[:rows]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        if rows == 1: 
            big_axes.set_title(row_titles[i], fontweight='semibold')
            big_axes.axis('off')
        else:
            big_axes[i].set_title(row_titles[i], fontweight='semibold')
            big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()
    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 1], states[step, :n_particle, 2])
                shapes._offsets3d = (states[step, n_particle:, 0], states[step, n_particle:, 1], states[step, n_particle:, 2])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)
    
    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=10))


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def update_position(n_shapes, prim_pos, positions=None, pts=None, floor=None, n_points=300, task_name='gripper'):
    if positions is None:
        positions = np.zeros([n_points + n_shapes, 3])
    if pts is not None:
        positions[:n_points, :3] = pts
    if floor is not None:
        positions[n_points, :3] = floor

    if task_name == 'gripper':
        positions[n_points+1, :3] = prim_pos[0]
        positions[n_points+2, :3] = prim_pos[1]
        # gt_positions[n_points+1, :3] = prim_pos1
        # gt_positions[n_points+2, :3] = prim_pos2
    else:
        positions[n_points+1, :3] = prim_pos[0]
        # gt_positions[n_points+1, :3] = prim_pos
    return positions


def shape_aug(states, n_points, gripper_h):
    states_tmp = states[:n_points]
    prim1 = states[n_points + 1]
    prim2 = states[n_points + 2]
    new_floor = np.array([[mid_point[0] - floor_size, mid_point[1] - floor_size, mid_point[2]], 
                        [mid_point[0] - floor_size, mid_point[1], mid_point[2]], 
                        [mid_point[0] - floor_size, mid_point[1] + floor_size, mid_point[2]],
                        [mid_point[0], mid_point[1] - floor_size, mid_point[2]], 
                        [mid_point[0], mid_point[1], mid_point[2]], 
                        [mid_point[0], mid_point[1] + floor_size, mid_point[2]],
                        [mid_point[0] + floor_size, mid_point[1] - floor_size, mid_point[2]], 
                        [mid_point[0] + floor_size, mid_point[1], mid_point[2]], 
                        [mid_point[0] + floor_size, mid_point[1] + floor_size, mid_point[2]]])
    
    n_gp = 11
    offset = gripper_h / (n_gp - 1)
    new_prim1 = []
    for j in range(n_gp):
        prim1_tmp = np.array([prim1[0], prim1[1], prim1[2] + offset * (j - (n_gp - 1) / 2)])
        new_prim1.append(prim1_tmp)
    new_prim1 = np.stack(new_prim1)

    new_prim2 = []
    for j in range(n_gp):
        prim2_tmp = np.array([prim2[0], prim2[1], prim2[2] + offset * (j - (n_gp - 1) / 2)])
        new_prim2.append(prim2_tmp)
    new_prim2 = np.stack(new_prim2)
    new_states = np.concatenate([states_tmp, new_floor, new_prim1, new_prim2])
    return new_states


def flip_inward_normals(pcd, center, threshold=0.7):
    # Flip normal if normal points inwards by changing vertex order
    # https://math.stackexchange.com/questions/3114932/determine-direction-of-normal-vector-of-convex-polyhedron-in-3d
    
    # Get vertices and triangles from the mesh
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # For each triangle in the mesh
    flipped_count = 0
    for i, n in enumerate(normals):
        # Compute vector from 1st vertex of triangle to center
        norm_ref = points[i] - center
        # Compare normal to the vector
        if np.dot(norm_ref, n) < 0:
            # Change vertex order to flip normal direction
            flipped_count += 1 
            if flipped_count > threshold * normals.shape[0]:
                normals = np.negative(normals)
                break

    pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    return pcd


def poisson_reconstruct_mesh_from_pcd_all(rest, cube, prim_pos, visualize=False):
    # raw_pcd = o3d.io.read_point_cloud(f"raw_pcd/gripper_ngrip_{frame}.ply")
    gripper_label = np.where(np.array(rest.colors)[:, 2] >= 0.6)
    grippers = rest.select_by_index(gripper_label[0])

    labels = np.array(grippers.cluster_dbscan(eps=0.03, min_points=100))
    gripper1 = grippers.select_by_index(np.where(labels == 0)[0])
    gripper2 = grippers.select_by_index(np.where(labels > 0)[0])
    for gripper in [gripper1, gripper2]:
        gripper.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        gripper.estimate_normals()
        gripper.orient_normals_consistent_tangent_plane(100)
        
        center = gripper.get_center()
        if np.dot(center - prim_pos[0], center - prim_pos[0]) < np.dot(center - prim_pos[1], center - prim_pos[1]):
            center = prim_pos[0]
        else:
            center = prim_pos[1]
        gripper = flip_inward_normals(gripper, center)

    cube.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    cube.estimate_normals()
    cube.orient_normals_consistent_tangent_plane(100)
    center = cube.get_center()
    cube = flip_inward_normals(cube, center)

    raw_pcd = gripper1 + gripper2 + cube

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(raw_pcd, depth=6)

    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
    
    return mesh


def reconstruct_mesh_from_pcd(pcd, algo="filter", alpha=0.5, depth=8, visualize=False):
    if algo == "filter":
        point_cloud = pv.PolyData(np.asarray(pcd.points))
        surf = point_cloud.reconstruct_surface()

        mf = pymeshfix.MeshFix(surf)
        mf.repair()
        pymesh = mf.mesh

        if visualize:
            pl = pv.Plotter()
            pl.add_mesh(point_cloud, color='k', point_size=10)
            pl.add_mesh(pymesh)
            pl.add_title('Reconstructed Surface')
            pl.show()

        mesh = pymesh
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(pymesh.points)
        # import pdb; pdb.set_trace()
        # mesh.triangles = o3d.utility.Vector3dVector(pymesh.faces.reshape(pymesh.n_faces, -1)[:, 1:])
    else:
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        center = pcd.get_center()
        pcd = flip_inward_normals(pcd, center)

        if algo == "ball_pivot":
            radii = [0.005, 0.01, 0.02, 0.04, 0.08]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        elif algo == "alpha_shape":
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha, tetra_mesh, pt_map)
        elif algo == "poisson":
            # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

        else:
            raise NotImplementedError

        # mesh.paint_uniform_color([0,1,0])
        mesh.compute_vertex_normals()

        if visualize:
            o3d_visualize([mesh])
    
    return mesh


def length(x_arr):
    return np.array([np.sqrt(x.dot(x) + 1e-8) for x in x_arr])


def is_inside(gripper_r, pt_pos, tool_pos, tool_rot, task="gripper"):
    if task=="gripper":
        pt_pos = pt_pos - np.tile(tool_pos, (pt_pos.shape[0], 1))
        pt_pos = (quat2mat(tool_rot) @ pt_pos.T).T
        p2 = copy.copy(pt_pos)
        # print(f"gp: {pt_pos}, pos: {position}, p2: {p2}")
        for i in range(p2.shape[0]):
            p2[i, 2] += gripper_h / 2 + 0.01
            p2[i, 2] -= min(max(p2[i, 2], 0.0), gripper_h)
        return length(p2) - gripper_r
    else:
        raise NotImplementedError


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def fps(pts, K=300):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def gen_data_one_frame_v1(rest, cube, prev_pcd, grippers, prim_pos, prim_rot, n_points, back, looking_ahead=5, visualize=False):
    cube_colors = np.asarray(cube.colors)
    color_avg = list(np.mean(cube_colors, axis=0))

    if not back:
        lower = cube.get_min_bound()
        upper = cube.get_max_bound()
        sample_size = round(5 * n_points)
        sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

        # selected_mesh = poisson_reconstruct_mesh_from_pcd_all(rest, cube, prim_pos, visualize=visualize)
        selected_mesh = reconstruct_mesh_from_pcd(rest, algo="poisson", depth=4, visualize=visualize)
        # f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])
        o3d.io.write_triangle_mesh("poisson.ply", selected_mesh)
        f = SDF(selected_mesh.vertices, selected_mesh.triangles)

        sdf = f(sampled_points)
        sampled_points = sampled_points[-sdf < 0, :]

        # is_touching = False
        # bounding_mesh = reconstruct_mesh_from_pcd(cube, algo='alpha_shape', alpha=0.5, visualize=visualize)
        # f = SDF(bounding_mesh.vertices, bounding_mesh.triangles)

        # for i, gripper_pcd in enumerate(grippers):
        #     gripper_points = np.asarray(gripper_pcd.points)
        #     sdf = f(gripper_points)
        #     gripper_points_in = gripper_points[-sdf < 0, :]
        #     print(f"Number of gripper points inside: {gripper_points_in.size}")
        #     if gripper_points_in.size > 0:
        #         is_touching = True
        #         break

        # if len(prev_pcd) >= looking_ahead and not is_touching:
        #     bounding_mesh = reconstruct_mesh_from_pcd(prev_pcd[-looking_ahead], algo='alpha_shape', alpha=0.005, visualize=visualize)
        #     # f = SDF(bounding_mesh.points, bounding_mesh.faces.reshape(bounding_mesh.n_faces, -1)[:, 1:])
        #     f = SDF(bounding_mesh.vertices, bounding_mesh.triangles)
        #     sdf = f(sampled_points)
        #     sampled_points = sampled_points[-sdf < 0, :]

        if visualize:
            sampled_pcd = o3d.geometry.PointCloud()
            sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
            sampled_pcd.paint_uniform_color(color_avg)
            o3d_visualize([sampled_pcd])

    if len(prev_pcd) > 0:
        # print(f'Touching status: {is_touching[0]} and {is_touching[1]}')
        prev_points = np.asarray(prev_pcd[-1].points)
        # sdf = f(prev_points)
        # prev_points = prev_points[-sdf < 0, :]
        if back:
            sampled_points = prev_points
        else:
            sampled_points = np.concatenate((sampled_points, prev_points))

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    if visualize:
        sampled_pcd.paint_uniform_color(color_avg)
        o3d_visualize([sampled_pcd])

    if not back:
        for tool_pos, tool_rot in zip(prim_pos, prim_rot):
            # if not in the tool, then it's valid
            inside_idx = is_inside(gripper_r, sampled_points, tool_pos, tool_rot)
            sampled_points = sampled_points[inside_idx > 0]  

        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_pcd = sampled_pcd.voxel_down_sample(voxel_size=0.002)
    
        if visualize:
            sampled_pcd.paint_uniform_color(color_avg)
            o3d_visualize([sampled_pcd])

        cl, inlier_ind = sampled_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        sampled_pcd = sampled_pcd.select_by_index(inlier_ind)
    
        if visualize:
            sampled_pcd.paint_uniform_color(color_avg)
            o3d_visualize([sampled_pcd])

    # selected_points = np.asarray(sampled_pcd.points)

    # selected_pcd = o3d.geometry.PointCloud()
    # selected_pcd.points = o3d.utility.Vector3dVector(selected_points)

    # cl, inlier_ind = selected_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    # selected_pcd = selected_pcd.select_by_index(inlier_ind)

    # if visualize:
    #     selected_pcd.paint_uniform_color([0,0,0])
    #     o3d.visualization.draw_geometries([selected_pcd])

    selected_points = fps(np.asarray(sampled_pcd.points), n_points)
    # selected_pcd.points = o3d.utility.Vector3dVector(selected_points)

    if visualize:
        fps_pcd = o3d.geometry.PointCloud()
        fps_pcd.points = o3d.utility.Vector3dVector(selected_points)
        fps_pcd.paint_uniform_color(color_avg)
        o3d_visualize([fps_pcd])

    return sampled_pcd, selected_points


# @profile
def gen_data_one_frame_v2(cube, prev_pcd, grippers, n_points, back, looking_ahead=5, surface=True, visualize=False):
    if visualize:
        cube.paint_uniform_color([0,0,0])
        o3d_visualize([cube])

    selected_points = np.asarray(cube.points)

    is_touching = [False, False]
    if not back:
        if len(prev_pcd) < looking_ahead:
            bounding_mesh = reconstruct_mesh_from_pcd(cube, algo='alpha_shape', alpha=0.5, visualize=visualize)
            f = SDF(bounding_mesh.vertices, bounding_mesh.triangles)
        else:
            curr_mesh = reconstruct_mesh_from_pcd(cube, algo='filter', visualize=visualize)
            f_curr = SDF(curr_mesh.points, curr_mesh.faces.reshape(curr_mesh.n_faces, -1)[:, 1:])

            prev_mesh = reconstruct_mesh_from_pcd(prev_pcd[-looking_ahead], algo='filter', visualize=visualize)
            f_prev = SDF(prev_mesh.points, prev_mesh.faces.reshape(prev_mesh.n_faces, -1)[:, 1:])

        for i, gripper_pcd in enumerate(grippers):
            gripper_points = np.asarray(gripper_pcd.points)
            if len(prev_pcd) < looking_ahead:
                sdf = f(gripper_points)
                gripper_points_in = gripper_points[-sdf < 0, :]
            else:
                sdf = f_curr(gripper_points)
                gripper_points_in = gripper_points[-sdf < 0, :]
                sdf = f_prev(gripper_points_in)
                gripper_points_in = gripper_points_in[-sdf < 0, :]
            print(f"Number of gripper points inside: {gripper_points_in.size}")
            if gripper_points_in.size > 0:
                is_touching[i] = True
                selected_points = np.concatenate((selected_points, gripper_points_in))

    if visualize:
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
        selected_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([selected_pcd])

    # import pdb; pdb.set_trace()
    # occulusion if not touching
    if len(prev_pcd) > 0 and not is_touching[0] and not is_touching[1]:
        # print(f'Touching status: {is_touching[0]} and {is_touching[1]}')
        prev_points = np.asarray(prev_pcd[-1].points)
        # sdf = f(prev_points)
        # prev_points = prev_points[-sdf < 0, :]
        selected_points = np.concatenate((selected_points, prev_points))

    if visualize:
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
        selected_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([selected_pcd])

    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
    selected_pcd = selected_pcd.voxel_down_sample(voxel_size=voxel_size)

    if not back:
        cl, inlier_ind = selected_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        selected_pcd = selected_pcd.select_by_index(inlier_ind)
        # cl, inlier_ind = selected_pcd.remove_radius_outlier(nb_points=10, radius=0.02)
        # selected_pcd = selected_pcd.select_by_index(inlier_ind)

    if visualize:
        selected_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([selected_pcd])

    if surface:
        selected_mesh = reconstruct_mesh_from_pcd(selected_pcd, algo='alpha_shape', alpha=0.03, visualize=visualize)
        selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(selected_mesh, n_points)
        if visualize:
            o3d_visualize([selected_surface])

        selected_points = np.asarray(selected_surface.points)
    else:
        lower = selected_pcd.get_min_bound()
        upper = selected_pcd.get_max_bound()
        sample_size = round(5 * n_points)
        sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower
        
        selected_mesh = reconstruct_mesh_from_pcd(selected_pcd, algo='filter', visualize=visualize)
        f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])

        # selected_mesh = reconstruct_mesh_from_pcd(selected_pcd, algo='poisson', depth=8, visualize=visualize)
        # f = SDF(selected_mesh.vertices, selected_mesh.triangles)
        sdf = f(sampled_points)
        sampled_points = sampled_points[-sdf < 0, :]

        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

        cl, inlier_ind = sampled_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        sampled_pcd = sampled_pcd.select_by_index(inlier_ind)

        selected_points = fps(np.asarray(sampled_pcd.points), K=n_points)

        if visualize:
            fps_pcd = o3d.geometry.PointCloud()
            fps_pcd.points = o3d.utility.Vector3dVector(selected_points)
            fps_pcd.paint_uniform_color([0,0,0])
            o3d_visualize([fps_pcd])

    return selected_pcd, selected_points


def merge_point_cloud(pcd_msgs, visualize=False):
    cloud_all_list = []
    for k in range(len(pcd_msgs)):
        cloud_rec = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msgs[k])
        cloud_array = cloud_rec.view('<f4').reshape(cloud_rec.shape + (-1,))
        points = cloud_array[:, :3]
        points = (quat2mat(depth_optical_frame_pose[3:]) @ points.T).T + depth_optical_frame_pose[:3]
        cam_ori = cam_pose_dict[f"cam_{k+1}"]["orientation"]
        points = (quat2mat(cam_ori) @ points.T).T + cam_pose_dict[f"cam_{k+1}"]["position"]
        
        cloud_rgb_bytes = cloud_array[:, -1].tobytes()
        # print(cloud_rgb_bytes)
        cloud_bgr = np.frombuffer(cloud_rgb_bytes, dtype=np.uint8).reshape(-1, 4) / 255 # int.from_bytes(cloud_rgb_bytes, 'big')
        cloud_rgb = cloud_bgr[:, ::-1]

        delta = 0.2
        x_filter = np.logical_and(points.T[0] > (mid_point[0] - delta), points.T[0] < (mid_point[0] + delta))
        y_filter = np.logical_and(points.T[1] > (mid_point[1] - delta), points.T[1] < (mid_point[1] + delta))
        points = points[np.logical_and(x_filter, y_filter)]
        cloud_rgb = cloud_rgb[np.logical_and(x_filter, y_filter)]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # print(cloud_rgb)
        pcd.colors = o3d.utility.Vector3dVector(cloud_rgb[:, 1:])

        cloud_all_list.append(pcd)

    cloud_all = o3d.geometry.PointCloud()
    for point_id in range(len(cloud_all_list)):
        # cloud_all_list[point_id].transform(pose_graph.nodes[point_id].pose)
        cloud_all += cloud_all_list[point_id]
    
    cl, inlier_ind = cloud_all.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    cloud_all = cloud_all.select_by_index(inlier_ind)

    cloud_all.voxel_down_sample(voxel_size=voxel_size)

    if visualize:
        o3d_visualize([cloud_all])

    # cloud_path = os.path.join(dataset_path, f'{vid_idx:03d}', f'pcd_{j:03d}.ply')
    # o3d.io.write_point_cloud(cloud_path, cloud_all)

    return cloud_all


def process_raw_pcd(pcd_all, visualize=False):
    # if visualize:
    #     o3d.visualization.draw_geometries([pcd_all])

    segment_models, inliers = pcd_all.segment_plane(distance_threshold=0.0075,ransac_n=3,num_iterations=100)
    rest = pcd_all.select_by_index(inliers, invert=True)

    lower = np.array([mid_point[0] - bounding_r, mid_point[1] - bounding_r, 0.005 + platform_h])
    upper = np.array([mid_point[0] + bounding_r, mid_point[1] + bounding_r, 0.05 + platform_h])
    rest = rest.crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))

    if visualize:
        o3d_visualize([rest])

    rest_colors = np.asarray(rest.colors)
    cube_label = np.where(np.logical_and(rest_colors[:, 0] < 0.1, rest_colors[:, 2] > 0.2))
    cube = rest.select_by_index(cube_label[0])
    grippers = rest.select_by_index(cube_label[0], invert=True)
    # gripper_colors = np.asarray(grippers.colors)
    # color_avg = list(np.mean(gripper_colors, axis=0))

    cl, inlier_ind = cube.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    cube = cube.select_by_index(inlier_ind)

    # cube.voxel_down_sample(voxel_size=voxel_size)

    if visualize:
        o3d_visualize([cube])

    n_bins = 30
    cube_points = np.asarray(cube.points)
    cube_colors = np.asarray(cube.colors)
    cube_z_hist_array = np.histogram(cube_points[:, 2], bins=n_bins)
    cube_z_count_max = np.argmax(cube_z_hist_array[0])
    cube_z_cap = cube_z_hist_array[1][min(cube_z_count_max + 4, n_bins - 1)]
    selected_idx = cube_points[:, 2] < cube_z_cap
    cube_points = cube_points[selected_idx]
    cube_colors = cube_colors[selected_idx]

    cube_new = o3d.geometry.PointCloud()
    cube_new.points = o3d.utility.Vector3dVector(cube_points)
    cube_new.colors = o3d.utility.Vector3dVector(cube_colors)

    cube_new_colors = np.asarray(cube_new.colors)
    color_avg = list(np.mean(cube_new_colors, axis=0))

    if visualize:
        cube_new.paint_uniform_color(color_avg)
        grippers.paint_uniform_color(color_avg)
        o3d_visualize([cube_new])

    cube_new.estimate_normals()
    cube_new.orient_normals_consistent_tangent_plane(100)

    rest = cube_new + grippers

    return rest, cube_new


def chamfer_distance(x, y):
    x = x[:, None, :].repeat(1, y.size(0), 1) # x: [N, M, D]
    y = y[None, :, :].repeat(x.size(0), 1, 1) # y: [N, M, D]
    dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
    dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
    dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

    return dis_xy + dis_yx


def em_distance(x, y):
    x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
    y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
    cost_matrix = dis.numpy()
    try:
        ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
    except:
        # pdb.set_trace()
        print("Error in linear sum assignment!")
    # print(f"EMD new_x shape: {new_x.shape}")
    # print(f"MAX: {torch.max(torch.norm(torch.add(new_x, -new_y), 2, dim=2))}")
    emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))
    
    return emd


n_instance = 1
gravity = 1
draw_mesh = 0
scene_params = np.zeros(3)
scene_params[0] = n_instance
scene_params[1] = gravity
scene_params[-1] = draw_mesh

depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]
ros_pkg_path = "/scr/hxu/catkin_ws/src/robocook_ros"
with open(os.path.join(ros_pkg_path, 'env', 'camera_pose_world.yml'), 'r') as f:
    cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

gripper_h = 0.075
gripper_r = 0.009
platform_h = 0.06

o3d_write = False
def main():
    visualize = False
    version = 0

    n_points = 300

    n_shapes = 3
    aug_n_shapes = 31
    data_names = ['positions', 'shape_quats', 'scene_params']
    floor_pos = mid_point
    floor_dim = 9
    primitive_dim = 11
    delta = 0.005

    task_name = 'ngrip_fixed_robot_3-29'
    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

    shape = 'X_square'

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    bag_path = os.path.join(ros_pkg_path, 'dataset', task_name, shape)
    
    bag_list = sorted(glob.glob(os.path.join(bag_path, '*.bag')), 
        key=lambda x:float(os.path.basename(x)[:-4]))
        # key=lambda x:float(os.path.basename(x)[-5]))

    rollout_path = os.path.join(cd, 'dump', 'robot_control_rebuttal', f'{shape}_replay')
    os.system('mkdir -p ' + rollout_path)

    shape_dir = os.path.join(os.getcwd(), 'shapes', 'alphabet_black', shape.split("_")[0])

    goal_frame_name = f'{shape.split("_")[0]}_robot.h5'
    goal_frame_path = os.path.join(shape_dir, goal_frame_name)
    goal_data = load_data(data_names, goal_frame_path)
    goal_shape = torch.FloatTensor(goal_data[0])[:n_points, :]

    sim_mean_p = np.array([0.49868582, 0.11530433, 0.49752659])
    sim_std_p = np.array([0.06167904, 0.05326168, 0.06180995])

    all_positions = []
    prev_pcd = []
    chamfer_loss_list = []
    emd_loss_list = []
    time_list = []

    back = False
    last_gripper_width = float('inf')
    for j in range(0, len(bag_list)):
        if j == len(bag_list) - 1:
            visualize = True
        print(f'+++++ Frame {j} +++++')
        bag = rosbag.Bag(bag_list[j])
        time_curr = float(os.path.basename(bag_list[j])[:-4])
        # time_curr = float(os.path.basename(bag_list[j])[-5])
        time_list.append(time_curr)

        pcd_msgs = []
        prim_pos = []
        prim_rot = []

        # --- get location info from topics --- #
        for topic, msg, t in bag.read_messages(
            topics=['/cam1/depth/color/points', '/cam2/depth/color/points', '/cam3/depth/color/points', '/cam4/depth/color/points', 
            '/gripper_1_pose', '/gripper_2_pose']
        ):
            if 'cam' in topic:
                pcd_msgs.append(msg)
            else:
                prim_pos.append(np.array([msg.position.x, msg.position.y, msg.position.z]))
                prim_rot.append(np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]))
        bag.close()

        # --- determine motion prior (grip or back) --- #
        gripper_width = np.linalg.norm(prim_pos[0] - prim_pos[1])
        if gripper_width > last_gripper_width + delta:
            back = True
        if back and gripper_width < last_gripper_width - delta:
            back = False
        if back: print(f"Moving back... {last_gripper_width} -> {gripper_width}")
        last_gripper_width = gripper_width

        pcd = merge_point_cloud(pcd_msgs, visualize=visualize)
        rest, cube = process_raw_pcd(pcd, visualize=visualize)

        grippers = []
        for k in range(len(prim_pos)):
            gripper = o3d.geometry.TriangleMesh.create_cylinder(gripper_r, gripper_h)
            gripper_pcd = gripper.sample_points_poisson_disk(500)
            gripper_pcd.paint_uniform_color([0,0,0])
            gripper_points = np.asarray(gripper_pcd.points)

            # NO NEED to do extra rotation
            gripper_points = (quat2mat(prim_rot[k]) @ euler2mat(0, 0, 0) @ gripper_points.T).T + prim_pos[k]
            gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points)
            grippers.append(gripper_pcd)

        if visualize:
            print("Visualize grippers...")
            o3d_visualize([cube, grippers[0], grippers[1]])

        if version == 0:
            selected_pcd, selected_points = gen_data_one_frame_v1(rest, cube, prev_pcd, grippers, prim_pos, prim_rot, n_points, back, visualize=visualize)
        # else:
        #     cube_points = np.asarray(cube.points)
        #     for tool_pos, tool_rot in zip(prim_pos, prim_rot):
        #         inside_idx = is_inside(gripper_r, cube_points, tool_pos, tool_rot)
        #         cube_points = cube_points[inside_idx > 0]

        #     cube_new = o3d.geometry.PointCloud()
        #     cube_new.points = o3d.utility.Vector3dVector(cube_points)
        #     selected_pcd, selected_points = gen_data_one_frame_v2(cube_new, prev_pcd, grippers, n_points, back, surface=False, visualize=visualize)
        
        prev_pcd.append(selected_pcd)

        o3d.io.write_point_cloud(os.path.join(rollout_path, f"{time_curr}.ply"), selected_pcd)

        if j >= 1:
            prev_positions = update_position(n_shapes, prim_pos, positions=prev_positions, n_points=n_points)
            prev_shape_positions = shape_aug(prev_positions, n_points, gripper_h)
            all_positions.append(prev_shape_positions)

            shape_shape_quats = np.zeros((aug_n_shapes, 4), dtype=np.float32)
            shape_shape_quats[floor_dim:floor_dim+primitive_dim] = prev_prim_ori1
            shape_shape_quats[floor_dim+primitive_dim:floor_dim+2*primitive_dim] = prev_prim_ori2

            shape_data = [prev_shape_positions, shape_shape_quats, scene_params]

            # store_data(data_names, shape_data, os.path.join(rollout_path, 'shape_' + str(j - 1) + '.h5'))

        positions = update_position(n_shapes, prim_pos, pts=selected_points, floor=floor_pos, n_points=n_points)
        
        prev_positions = positions
        prev_prim_ori1 = prim_rot[0]
        prev_prim_ori2 = prim_rot[1]

        if j == 0:
            # self.mean_p = np.array([0.437, 0.0, 0.0725])
            std_p = np.array([0.017, 0.017, 0.006])
            mean_p = np.mean(selected_points[:n_points], axis=0)
            # self.std_p = np.std(selected_points[:self.n_particle], axis=0)
            print(f"The first frame: {mean_p} +- {std_p}")

        selected_points = (selected_points - mean_p) / std_p
        selected_points = np.array([selected_points.T[0], selected_points.T[2], selected_points.T[1]]).T \
            * np.array([0.06, sim_std_p[1], 0.06]) + np.array([0.5, sim_mean_p[1], 0.5])

        shape_curr = torch.FloatTensor(selected_points)
        chamfer_loss = chamfer_distance(shape_curr, goal_shape)
        emd_loss = em_distance(shape_curr, goal_shape)
        chamfer_loss_list.append(chamfer_loss)
        emd_loss_list.append(emd_loss)
        print(f'chamfer: {chamfer_loss}, emd: {emd_loss}')

    eval_plot_curves(time_list, (chamfer_loss_list, emd_loss_list), os.path.join(rollout_path, 'loss.png'))
    print([round(x.item(), 4) for x in chamfer_loss_list])
    print([round(x.item(), 4) for x in emd_loss_list])
    plt_render([np.array(all_positions)], n_points, os.path.join(rollout_path, 'plt.gif'))

if __name__ == '__main__':
    main()
