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
import string
import sys
import torch
import trimesh
import yaml

from config import gen_args
from datetime import datetime
from data_utils import store_data, load_data, get_scene_info, get_env_group, prepare_input
from pysdf import SDF
from sensor_msgs.msg import PointCloud2
from sys import platform
from timeit import default_timer as timer
from transforms3d.euler import euler2mat
from transforms3d.quaternions import *

from plb.engine.taichi_env import TaichiEnv
from plb.config import load


task_params = {
    "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
    "sample_radius": 0.4,
    "len_per_grip": 30,
    "len_per_grip_back": 10,
    "floor_pos": np.array([0.5, 0, 0.5]),
    "n_shapes": 3, 
    "n_shapes_floor": 9,
    "n_shapes_per_gripper": 11,
    "gripper_mid_pt": int((11 - 1) / 2),
    "gripper_gap_limits": np.array([0.14, 0.06]), # ((0.4 * 2 - (0.23)) / (2 * 30), (0.4 * 2 - 0.15) / (2 * 30)),
    "p_noise_scale": 0.08,
    "p_noise_bound": 0.1,
    "loss_weights": [0.9, 0.1, 0.0, 0.0],
    "tool_size": 0.045
}


def o3d_visualize(display_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in display_list:
        vis.add_geometry(geo)
        vis.update_geometry(geo)
    
    # vis.get_render_option().light_on = False
    vis.get_render_option().point_size = 10
    vis.get_render_option().mesh_show_back_face = True
    # vis.get_render_option().mesh_show_wireframe = True

    vis.poll_events()
    vis.update_renderer()

    if o3d_write:
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        image_path = os.path.join(cd, '..', '..', 'images', f'{time_now}.png')
        print(image_path)
        vis.capture_screen_image(image_path)
        vis.destroy_window()
    else:
        vis.run()
        # o3d.visualization.draw_geometries(display_list, mesh_show_back_face=True)

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

def eval_plot_iou_curve(time_list, loss_list, path=''):
    plt.figure(figsize=[16, 9])
    loss_iou = list(zip(*loss_list))
    for i in range(len(loss_iou)):
        plt.plot(time_list, loss_iou[i], linewidth=6, label=f'IOU_{voxel_size}')
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

def visualize_points(ax, all_points, n_points):
    points = ax.scatter(all_points[:n_points, 0], all_points[:n_points, 2], all_points[:n_points, 1], c='b', s=10)
    shapes = ax.scatter(all_points[n_points:, 0], all_points[n_points:, 2], all_points[n_points:, 1], c='r', s=10)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # for ctr, dim in zip(centers, 'xyz'):
    #     getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.invert_yaxis()

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
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)
    
    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=10))


def plt_render_frames_rm(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    # pdb.set_trace()
    n_frames = particles_set[0].shape[0]
    rows = 1
    cols = 1

    fig, big_axes = plt.subplots(rows, 1, figsize=(3, 9))
    row_titles = ['Prediction']
    views = [(90, 90)]
    plot_info_all = {}
    for i in range(rows):
        states = particles_set[i]
        big_axes.set_title(row_titles[i], fontweight='semibold')
        big_axes.axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.axis('off')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    for step in range(n_frames): # n_frames
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle+9:, 0], states[step, n_particle+9:, 2], states[step, n_particle+9:, 1])

        plt.tight_layout()
        plt.savefig(f'{render_path}/{str(step).zfill(3)}.pdf')


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


def poisson_reconstruct_mesh(cube, visualize=False):
    cube.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    cube.estimate_normals()
    cube.orient_normals_consistent_tangent_plane(100)
    # center = cube.get_center()
    # cube = flip_inward_normals(cube, center)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cube, depth=4)

    if visualize:
        mesh.paint_uniform_color([0.2, 0.5, 0.2])
        o3d.visualization.draw_geometries([cube, mesh], mesh_show_back_face=True, point_show_normal=True)
    
    return mesh


def reconstruct_mesh_from_pcd(pcd, alpha=0.05, visualize=False):
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)

    if visualize:
        mesh.paint_uniform_color([0.2, 0.5, 0.2])
        o3d.visualization.draw_geometries([pcd, mesh], 
            mesh_show_back_face=True, mesh_show_wireframe=True)

    return mesh


def voxelize(points, sample_size=20000, visualize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    lower = pcd.get_min_bound()
    upper = pcd.get_max_bound()
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    mesh = reconstruct_mesh_from_pcd(pcd, visualize=False)
    f = SDF(mesh.vertices, mesh.triangles)

    sdf = f(sampled_points)
    sampled_points = sampled_points[-sdf < 0, :]
    
    if visualize:
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

        vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(sampled_pcd, voxel_size, lower, upper)
        o3d_visualize([vg])

    return sampled_points


def iou(x, y):
    # x: [N, D]
    # y: [N, D]

    # import pdb; pdb.set_trace()
    N = x.shape[0]
    xy = np.concatenate([x, y], axis=0)
    # min_bound = np.floor(np.min(xy, axis=0))
    # max_bound = np.ceil(np.max(xy, axis=0))
    min_bound = np.floor(np.min(xy, axis=0) * 10) / 10
    max_bound = np.ceil(np.max(xy, axis=0) * 10) / 10

    # x_pcd = o3d.geometry.PointCloud()
    # x_pcd.points = o3d.utility.Vector3dVector(x)

    # y_pcd = o3d.geometry.PointCloud()
    # y_pcd.points = o3d.utility.Vector3dVector(y)

    # x_pcd.paint_uniform_color([0.5, 0.2, 0.2])
    # y_pcd.paint_uniform_color([0.2, 0.2, 0.5])

    # x_vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(x_pcd, voxel_size, min_bound, max_bound)
    # y_vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(y_pcd, voxel_size, min_bound, max_bound)
    
    # o3d_visualize([x_vg, y_vg])

    grid_x = np.zeros([int((max_bound[0]-min_bound[0])/voxel_size)+1,
                        int((max_bound[1]-min_bound[1])/voxel_size)+1,
                        int((max_bound[2]-min_bound[2])/voxel_size)+1])

    grid_y = np.zeros_like(grid_x)
    # import pdb; pdb.set_trace()
    for i in range(x.shape[0]):
        x1 = int(np.floor((x[i][0] - min_bound[0]) / voxel_size))
        x2 = int(np.floor((x[i][1] - min_bound[1]) / voxel_size))
        x3 = int(np.floor((x[i][2] - min_bound[2]) / voxel_size))
        grid_x[x1, x2, x3] = 1
        # print('x1x2x3', x1, x2, x3)
    
    for i in range(y.shape[0]):
        y1 = int(np.floor((y[i][0] - min_bound[0]) / voxel_size))
        y2 = int(np.floor((y[i][1] - min_bound[1]) / voxel_size))
        y3 = int(np.floor((y[i][2] - min_bound[2]) / voxel_size))
        grid_y[y1, y2, y3] = 1
        # print('y1y2y3', y1, y2, y3)

    intersection = grid_x * grid_y
    union = grid_x + grid_y - grid_x * grid_y
    iou = np.sum(intersection)/np.sum(union)
    return iou


n_instance = 1
gravity = 1
draw_mesh = 0
scene_params = np.zeros(3)
scene_params[0] = n_instance
scene_params[1] = gravity
scene_params[-1] = draw_mesh

voxel_size = 0.01
o3d_write = False

def main():
    visualize = False
    n_points = 300

    n_shapes = 3
    aug_n_shapes = 31
    data_names = ['positions', 'shape_quats', 'scene_params']
    floor_dim = 9
    primitive_dim = 11
    delta = 0.005

    task_name = 'ngrip_fixed_robot_3-29'
    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

     # set up the env
    cfg = load("../PlasticineLab/plb/envs/gripper_fixed.yml")
    print(cfg)

    env = None
    state = None
    if platform != 'darwin':
        env = TaichiEnv(cfg, nn=False, loss=False)
        env.initialize()
        state = env.get_state()

        env.set_state(**state)
        taichi_env = env

        env.renderer.camera_pos[0] = 0.5
        env.renderer.camera_pos[1] = 2.5
        env.renderer.camera_pos[2] = 0.5
        env.renderer.camera_rot = (1.57, 0.0)

        env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
        env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])
        
        def set_parameters(env: TaichiEnv, yield_stress, E, nu):
            env.simulator.yield_stress.fill(yield_stress)
            _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
            env.simulator.mu.fill(_mu)
            env.simulator.lam.fill(_lam)

        set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2

        def update_camera(env):
            env.renderer.camera_pos[0] = 0.5 #np.array([float(i) for i in (0.5, 2.5, 0.5)]) #(0.5, 2.5, 0.5)  #.from_numpy(np.array([[0.5, 2.5, 0.5]]))
            env.renderer.camera_pos[1] = 2.5
            env.renderer.camera_pos[2] = 2.2
            env.renderer.camera_rot = (0.8, 0.0)
            env.render_cfg.defrost()
            env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.2)
            env.render_cfg.camera_rot_1 = (0.8, 0.)
            env.render_cfg.camera_pos_2 = (2.4, 2.5, 0.2)
            env.render_cfg.camera_rot_2 = (0.8, 1.8)
            env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0.2)
            env.render_cfg.camera_rot_3 = (0.8, -1.8)
            env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
            env.render_cfg.camera_rot_4 = (0.8, 3.14)

        update_camera(env)

        # update the tool size
        env.primitives.primitives[0].r[None] = task_params["tool_size"]
        env.primitives.primitives[1].r[None] = task_params["tool_size"]

    # shape = 'A'

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))

    # for shape in string.ascii_uppercase:
    for shape in ['X']:
        result_path = "/scr/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/dump_ngrip_fixed_robot_v4" \
            + "/files_dy_21-Jan-2022-22:33:11.729243_nHis4_aug0.05_gt0_seqlen6_emd0.9_chamfer0.1_uh0.0_clip0.0" \
            + "/control_robot/X/sim_0+algo_fix+3_grips+CEM+emd_chamfer_uh_clip+correction_0+debug_0"
        # result_path = os.path.join(cd, 'dump', 'sim_control_final', shape, 'selected')
        init_pose_seq_opt = np.load(os.path.join(result_path, 'init_pose_seq_opt.npy'))
        act_seq_opt = np.load(os.path.join(result_path, 'act_seq_opt.npy'))

        rollout_path = os.path.join(cd, 'dump', 'sim_control_rebuttal', f'{shape}_FEM_replay')
        os.system('mkdir -p ' + rollout_path)

        shape_dir = os.path.join(cd, 'shapes', 'alphabet_bold', shape.split("_")[0])

        goal_frame_name = f'{shape.split("_")[0]}.h5'
        goal_frame_path = os.path.join(shape_dir, goal_frame_name)
        goal_data = load_data(data_names, goal_frame_path)
        goal_shape = torch.FloatTensor(goal_data[0])[:n_points, :]

        goal_shape_upsample = voxelize(goal_shape, visualize=False)

        all_positions = []
        chamfer_loss_list = []
        emd_loss_list = []
        iou_loss_list = []
        time_list = []

        # import pdb; pdb.set_trace()
        taichi_env.set_state(**state)
        for i in range(act_seq_opt.shape[0]):
            taichi_env.primitives.primitives[0].set_state(0, init_pose_seq_opt[i, task_params["gripper_mid_pt"], :7])
            taichi_env.primitives.primitives[1].set_state(0, init_pose_seq_opt[i, task_params["gripper_mid_pt"], 7:])
            for j in range(act_seq_opt.shape[1]):
                step = i * act_seq_opt.shape[1] + j

                taichi_env.step(act_seq_opt[i][j])
                x = taichi_env.simulator.get_x(0)
                step_size = len(x) // n_points
                # print(f"x before: {x.shape}")
                x = x[::step_size]
                particles = x[:n_points]
                # print(f"x after: {x.shape}")

                shape_curr = torch.FloatTensor(particles)
                # if step > 100:
                #     visualize = True
                shape_curr_upsample = voxelize(shape_curr, visualize=visualize)
                chamfer_loss = chamfer_distance(shape_curr, goal_shape)
                emd_loss = em_distance(shape_curr, goal_shape)
                iou_loss_slice = []
                # for voxel_size in voxel_size_list:
                iou_loss = iou(shape_curr_upsample, goal_shape_upsample)
                iou_loss_slice.append(iou_loss)

                time_list.append(step)
                chamfer_loss_list.append(chamfer_loss)
                emd_loss_list.append(emd_loss)
                iou_loss_list.append(iou_loss_slice)
                all_positions.append(particles)
                print(f'chamfer: {chamfer_loss}, emd: {emd_loss}, iou: {iou_loss_slice}')

        eval_plot_curves(time_list, (chamfer_loss_list, emd_loss_list), os.path.join(rollout_path, 'loss.png'))
        eval_plot_iou_curve(time_list, iou_loss_list, os.path.join(rollout_path, 'loss_iou.png'))
        print([round(x.item(), 4) for x in chamfer_loss_list])
        print([round(x.item(), 4) for x in emd_loss_list])
        # print([round(x.item(), 4) for x in iou_loss_list])
        plt_render([np.array(all_positions)], n_points, os.path.join(rollout_path, 'plt.gif'))
        plt_render_frames_rm([np.array(all_positions)], n_points, rollout_path)


if __name__ == '__main__':
    main()
