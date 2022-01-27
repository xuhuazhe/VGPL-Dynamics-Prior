import copy
import os
import numpy as np
import imageio

from config import gen_args
from data_utils import load_data
from utils import create_instance_colors, set_seed,  Tee, count_parameters

import pdb
from plb.engine.taichi_env import TaichiEnv
from plb.config import load
from plb.algorithms import sample_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import taichi as ti
import glob
ti.init(arch=ti.gpu)

task_params = {
    "mid_point": np.array([0.5, 0.4, 0.5, 0, 0, 0]),
    "default_h": 0.14,
    "sample_radius": 0.25,
    "n_grips": 3,
    "gripper_rate": 0.01,
    "len_per_grip": 30,
    "len_per_grip_back": 10,
    "floor_pos": np.array([0.5, 0, 0.5]),
    "n_shapes": 3, 
    "n_shapes_floor": 9,
    "n_shapes_per_gripper": 11,
    "gripper_mid_pt": int((11 - 1) / 2),
    "tool_size_small": 0.03,
    "tool_size_large": 0.045,
}


def visualize_points_helper(ax, all_points, n_particles, p_color='b', alpha=1.0):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c=p_color, s=10)
    shapes = ax.scatter(all_points[n_particles+9:, 0], all_points[n_particles+9:, 2], all_points[n_particles+9:, 1], c='r', s=20)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # ax.invert_yaxis()

    return points, shapes


def plt_render_frames(particles_set, target_shape, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = 1
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 3))
    # plt.gca().invert_yaxis()
    row_titles = ['Simulator']
    # views = [(90, 90)]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        states = particles_set[i]
        if rows == 1:
            big_axes.set_title(row_titles[i], fontweight='semibold')
            big_axes.axis('off')
        else:  
            big_axes[i].set_title(row_titles[i], fontweight='semibold')
            big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.axis('off')
            ax.view_init(*views[j])
            visualize_points_helper(ax, target_shape, n_particle, p_color='c', alpha=1.0)
            points, shapes = visualize_points_helper(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    frame_list = [n_frames-1]
    # for g in range(n_frames // (task_params['len_per_grip'] + task_params['len_per_grip_back'])):
    #     frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + 12)
    #     frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + 15)
    #     frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + task_params['len_per_grip'] - 1)

    for step in frame_list: # range(n_frames):
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle+9:, 0], states[step, n_particle+9:, 2], states[step, n_particle+9:, 1])

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{render_path}/{str(step).zfill(3)}.pdf')


# def sample_particles(env, cam_params, k_fps_particles, n_particles=2000):
#     prim_pos1 = env.primitives.primitives[0].get_state(0)
#     prim_pos2 = env.primitives.primitives[1].get_state(0)
#     prim_pos = [prim_pos1[:3], prim_pos2[:3]]
#     prim_rot = [prim_pos1[3:], prim_pos2[3:]]

#     img = env.render_multi(mode='rgb_array', spp=3)
#     rgb, depth = img[0], img[1]

#     tool_info = {'tool_size': task_params["tool_size"]}

#     sampled_points = sample_data.gen_data_one_frame(rgb, depth, cam_params, prim_pos, prim_rot, n_particles, k_fps_particles, tool_info)

#     positions = sample_data.update_position(task_params["n_shapes"], prim_pos, pts=sampled_points, 
#                                             floor=task_params["floor_pos"], k_fps_particles=k_fps_particles)
#     shape_positions = sample_data.shape_aug(positions, k_fps_particles)

#     return shape_positions


def add_shapes(state_seq, init_pose_seq, act_seq, k_fps_particles, mode):
    updated_state_seq = []
    for i in range(act_seq.shape[0]):
        prim_pos1 = copy.copy(init_pose_seq[i, task_params["gripper_mid_pt"], :3])
        prim_pos2 = copy.copy(init_pose_seq[i, task_params["gripper_mid_pt"], 7:10])
        prim_rot1 = copy.copy(init_pose_seq[i, task_params["gripper_mid_pt"], 3:7])
        prim_rot2 = copy.copy(init_pose_seq[i, task_params["gripper_mid_pt"], 10:])
        for j in range(act_seq.shape[1]):
            idx = i * act_seq.shape[1] + j
            prim_pos1 += 0.02 * act_seq[i, j, :3]
            prim_pos2 += 0.02 * act_seq[i, j, 6:9]
            positions = sample_data.update_position(task_params["n_shapes"], [prim_pos1, prim_pos2], pts=state_seq[idx], 
                                                    floor=task_params["floor_pos"], k_fps_particles=k_fps_particles)
            if mode == '2d':
                shape_positions = sample_data.shape_aug(positions, k_fps_particles)
            else:
                shape_positions = sample_data.shape_aug_3D(positions, prim_rot1, prim_rot2, k_fps_particles)
            updated_state_seq.append(shape_positions)
    return np.stack(updated_state_seq)


def main():
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_control) > 0:
        args.outf = args.outf_control

    n_particle = 300

    if len(args.goal_shape_name) > 0 and args.goal_shape_name != 'none':
        vid_idx = 0
        if args.goal_shape_name[:3] == 'vid':
            vid_idx = int(args.goal_shape_name[4:])
            shape_goal_dir = str(vid_idx).zfill(3)
        else:
            shape_goal_dir = args.goal_shape_name
    else:
        print("Please specify a valid goal shape name!")
        raise ValueError

    rollout_dir = f"./data/data_{args.data_type}/train/"
    cam_params = np.load(rollout_dir + "../cam_params.npy", allow_pickle=True)

    # load goal shape
    data_names = ['positions', 'shape_quats', 'scene_params']
    if len(args.goal_shape_name) > 0 and args.goal_shape_name != 'none' and args.goal_shape_name[:3] != 'vid':
        # if len(args.goal_shape_name) > 1:
        #     shape_type = 'simple'
        # else:
        #     shape_type = "alphabet"
        shape_dir = os.path.join(os.getcwd(), 'shapes', args.shape_type, args.goal_shape_name)
        goal_frame_name = f'{args.goal_shape_name}.h5'
        # if args.shape_aug:
        #     goal_frame_name = 'shape_' + goal_frame_name
        goal_frame_path = os.path.join(shape_dir, goal_frame_name)
        goal_data = load_data(data_names, goal_frame_path)
        goal_shape = goal_data[0][:n_particle, :]
    else:
        frame_path = os.path.join(rollout_dir, str(vid_idx).zfill(3), f'shape_118.h5') 
        this_data = load_data(data_names, frame_path)
        goal_shape = this_data[0][:n_particle, :]

    control_out_dir = args.outf

    # set up the env
    cfg = load(args.gripperf)
    print(cfg)

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
        env.renderer.camera_pos[0] = 0.5
        env.renderer.camera_pos[1] = 2.2
        env.renderer.camera_pos[2] = 0.5
        env.renderer.camera_rot = (np.pi/2, 0.0)
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

    small_list = 'EFMNZ'
    try:
        init_pose_seq = np.load(f"{control_out_dir}/init_pose_seq_opt.npy", allow_pickle=True)
        act_seq = np.load(f"{control_out_dir}/act_seq_opt.npy", allow_pickle=True)

        if os.path.exists(f"{control_out_dir}/tool_seq_opt.npy"):
            tool_seq = np.load(f"{control_out_dir}/tool_seq_opt.npy", allow_pickle=True)
            tool_seq = np.concatenate([tool_seq, tool_seq[-1:, :, :]], axis=0)
        else:
            if args.goal_shape_name in small_list and \
                    not (args.goal_shape_name == 'K' and 'regular' not in  control_out_dir):
                tool_seq = np.zeros([act_seq.shape[0], 1, 1])
            else:
                tool_seq = np.ones([act_seq.shape[0], 1, 1])
    except:
        print('opt not found')
        # init_pose_seq = np.load(f"{control_out_dir}/init_pose_seq_{str(2)}.npy", allow_pickle=True)
        # act_seq = np.load(f"{control_out_dir}/act_seq_{str(2)}.npy", allow_pickle=True)
        # if os.path.exists(f"{control_out_dir}/tool_seq_{str(2)}.npy"):
        #     tool_seq = np.load(f"{control_out_dir}/tool_seq_{str(2)}.npy", allow_pickle=True)
        # else:
        #     if args.goal_shape_name in small_list:
        #         tool_seq = np.zeros([act_seq.shape[0], 1, 1])
        #     else:
        #         tool_seq = np.ones([act_seq.shape[0], 1, 1])

    # files = glob.glob(control_out_dir+'/*_rgb.png')
    # for f in files:
    #     os.remove(f)
    # files = glob.glob(control_out_dir + '/*.mp4')
    # for f in files:
    #     os.remove(f)

    print(init_pose_seq.shape, act_seq.shape)
    if args.goal_shape_name == 'D':
        init_pose_seq = init_pose_seq[1:, :, :]
        act_seq = act_seq[1:, :, :]
        tool_seq = tool_seq[1:, :, :]
    elif args.goal_shape_name == 'E' and 'tool' in control_out_dir:
        init_pose_seq = init_pose_seq[:-2, :, :]
        act_seq = act_seq[:-2, :, :]
        tool_seq = tool_seq[:-2, :, :]
    # elif args.goal_shape_name == 'O' and 'tool' not in control_out_dir:
    #     init_pose_seq = init_pose_seq[:1, :, :]
    #     act_seq = np.zeros_like(act_seq[:1, :, :])
    #     tool_seq = tool_seq[:1, :, :]
    elif args.goal_shape_name == 'T':
        init_pose_seq = init_pose_seq[:-1, :, :]
        act_seq = act_seq[:-1, :, :]
        tool_seq = tool_seq[:-1, :, :]

    env.set_state(**state)
    state_seq = []
    for i in range(act_seq.shape[0]):
        # if args.goal_shape_name == 'D' and i == 0:
        #     import pdb; pdb.set_trace()
        #     continue
        if tool_seq[i, 0, 0] == 1:
            env.primitives.primitives[0].r = task_params['tool_size_large']
            env.primitives.primitives[1].r = task_params['tool_size_large']
        else:
            env.primitives.primitives[0].r = task_params['tool_size_small']
            env.primitives.primitives[1].r = task_params['tool_size_small']
        env.primitives.primitives[0].set_state(0, init_pose_seq[i, task_params["gripper_mid_pt"], :7])
        env.primitives.primitives[1].set_state(0, init_pose_seq[i, task_params["gripper_mid_pt"], 7:])
        for j in range(act_seq.shape[1]):
            true_idx = i * act_seq.shape[1] + j
            env.step(act_seq[i][j])
            x = env.simulator.get_x(0)
            step_size = len(x) // n_particle
            # print(f"x before: {x.shape}")
            x = x[::step_size]
            particles = x[:n_particle]
            # print(f"x after: {x.shape}")
            state_seq.append(particles)
            # rgb_img, depth_img = env.render(mode='get')
            # import pdb; pdb.set_trace()
            # rgb_img = np.flip(rgb_img, 0)
            # rgb_img = np.flip(rgb_img, 1)
            # imageio.imwrite(f"{control_out_dir}/{true_idx:03d}_rgb.png", rgb_img)

    # pdb.set_trace()
    sim_state_seq = add_shapes(state_seq, init_pose_seq, act_seq, n_particle, mode='3d')
    plt_render_frames([sim_state_seq], goal_shape, n_particle, control_out_dir)
    # os.system(
    #     f'ffmpeg -y -i {control_out_dir}/%03d_rgb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {control_out_dir}/vid000.mp4')


if __name__ == "__main__":
    main()