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

def main():
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

    for shape in string.ascii_uppercase:
        result_path = os.path.join(cd, 'dump', 'sim_control_final', shape, 'selected')
        init_pose_seq_opt = np.load(os.path.join(result_path, 'init_pose_seq_opt.npy'))
        act_seq_opt = np.load(os.path.join(result_path, 'act_seq_opt.npy'))

        rollout_path = os.path.join(cd, 'dump', 'sim_control_rebuttal', f'{shape}_replay')
        os.system('mkdir -p ' + rollout_path)

        shape_dir = os.path.join(cd, 'shapes', 'alphabet_bold', shape.split("_")[0])

        goal_frame_name = f'{shape.split("_")[0]}.h5'
        goal_frame_path = os.path.join(shape_dir, goal_frame_name)
        goal_data = load_data(data_names, goal_frame_path)
        goal_shape = torch.FloatTensor(goal_data[0])[:n_points, :]

        all_positions = []
        chamfer_loss_list = []
        emd_loss_list = []
        time_list = []

        # import pdb; pdb.set_trace()
        taichi_env.set_state(**state)
        for i in range(act_seq_opt.shape[0]):
            taichi_env.primitives.primitives[0].set_state(0, init_pose_seq_opt[i, task_params["gripper_mid_pt"], :7])
            taichi_env.primitives.primitives[1].set_state(0, init_pose_seq_opt[i, task_params["gripper_mid_pt"], 7:])
            for j in range(act_seq_opt.shape[1]):
                taichi_env.step(act_seq_opt[i][j])
                x = taichi_env.simulator.get_x(0)
                step_size = len(x) // n_points
                # print(f"x before: {x.shape}")
                x = x[::step_size]
                particles = x[:n_points]
                # print(f"x after: {x.shape}")

                shape_curr = torch.FloatTensor(particles)
                chamfer_loss = chamfer_distance(shape_curr, goal_shape)
                emd_loss = em_distance(shape_curr, goal_shape)

                time_list.append(i * act_seq_opt.shape[1] + j)
                chamfer_loss_list.append(chamfer_loss)
                emd_loss_list.append(emd_loss)
                all_positions.append(particles)
                print(f'chamfer: {chamfer_loss}, emd: {emd_loss}')

        eval_plot_curves(time_list, (chamfer_loss_list, emd_loss_list), os.path.join(rollout_path, 'loss.png'))
        print([round(x.item(), 4) for x in chamfer_loss_list])
        print([round(x.item(), 4) for x in emd_loss_list])
        plt_render([np.array(all_positions)], n_points, os.path.join(rollout_path, 'plt.gif'))


if __name__ == '__main__':
    main()
