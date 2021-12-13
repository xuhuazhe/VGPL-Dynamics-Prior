import os
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange
import copy
import imageio

from matplotlib import cm
import matplotlib.pyplot as plt
import pdb

from config import gen_args
from data_utils import load_data, get_scene_info, get_env_group, prepare_input
from models import Model, EarthMoverLoss, L1ShapeLoss
from utils import create_instance_colors, set_seed,  Tee, count_parameters

from plb.engine.taichi_env import TaichiEnv
from plb.config import load
from plb.algorithms import sample_data

from sys import platform
import gc

import taichi as ti
ti.init(arch=ti.cuda)

task_params = {
    "mid_point": np.array([0.5, 0.4, 0.5, 0, 0, 0]),
    "default_h": 0.14,
    "sample_radius": 0.25,
    "n_grips": 3,
    "gripper_rate": 0.01,
    "len_per_grip": 20,
    "len_per_grip_back": 10,
    "n_shapes_floor": 9,
    "n_shapes_per_gripper": 11,
    "gripper_mid_pt": int((11 - 1) / 2)
}

def main():
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_control) > 0:
        args.outf = args.outf_control

    if args.gt_action:
        test_name = f'sim_{args.use_sim}+gt_action_{args.gt_action}+{args.reward_type}'
    else:
        test_name = f'sim_{args.use_sim}+algo_{args.control_algo}+{args.n_grips}_grips+{args.opt_algo}+{args.reward_type}+correction_{args.correction}+debug_{args.debug}'

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

    control_out_dir = os.path.join(args.outf, 'control', shape_goal_dir, test_name)
    print(control_out_dir)

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

    init_pose_seq = np.load(f"{control_out_dir}/init_pose_seq_opt.npy", allow_pickle=True)
    act_seq = np.load(f"{control_out_dir}/act_seq_opt.npy", allow_pickle=True)
    print(init_pose_seq.shape, act_seq.shape)

    for i in range(act_seq.shape[0]):
        env.primitives.primitives[0].set_state(0, init_pose_seq[i, task_params["gripper_mid_pt"], :7])
        env.primitives.primitives[1].set_state(0, init_pose_seq[i, task_params["gripper_mid_pt"], 7:])
        for j in range(act_seq.shape[1]):
            true_idx = i * act_seq.shape[1] + j
            env.step(act_seq[i][j])
            rgb_img, depth_img = env.render(mode='get')
            imageio.imwrite(f"{control_out_dir}/{true_idx:03d}_rgb.png", rgb_img)

    os.system(f'ffmpeg -y -i {control_out_dir}/%03d_rgb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {control_out_dir}/vid000.mp4')


if __name__ == "__main__":
    main()