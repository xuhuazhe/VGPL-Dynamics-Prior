import os
import numpy as np
import imageio


from config import gen_args
from utils import create_instance_colors, set_seed,  Tee, count_parameters

from plb.engine.taichi_env import TaichiEnv
from plb.config import load

import taichi as ti
import glob
ti.init(arch=ti.gpu)

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
    "gripper_mid_pt": int((11 - 1) / 2),
    "tool_size_small": 0.03,
    "tool_size_large": 0.045,
}

def main():
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_control) > 0:
        args.outf = args.outf_control

    # if args.gt_action:
    #     test_name = f'sim_{args.use_sim}+gt_action_{args.gt_action}+{args.reward_type}'
    # else:
    #     test_name = f'sim_{args.use_sim}+algo_{args.control_algo}+{args.n_grips}_grips+{args.opt_algo}+{args.reward_type}+correction_{args.correction}+debug_{args.debug}'

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

    parent_dir = os.path.join(args.outf, 'sim_control_final', shape_goal_dir)
    all_dirs = os.listdir(parent_dir)
    control_out_list = []
    for i in range(len(all_dirs)):
        # if 'max' in sorted(all_dirs)[i]:
        control_out_list.append(os.path.join(parent_dir, sorted(all_dirs)[i]))
    # print(control_out_dir)
    chosen_appendix = 'opt'
    # set up the env
    cfg = load(args.gripperf)
    print(cfg)

    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()

    for index, control_out_dir in enumerate(control_out_list):
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
        small_list = 'EFKLMNSWZ'
        try:
            init_pose_seq = np.load(f"{control_out_dir}/init_pose_seq_{str(chosen_appendix)}.npy", allow_pickle=True)
            act_seq = np.load(f"{control_out_dir}/act_seq_{str(chosen_appendix)}.npy", allow_pickle=True)
            if os.path.exists(f"{control_out_dir}/tool_seq_{str(chosen_appendix)}.npy"):
                tool_seq = np.load(f"{control_out_dir}/tool_seq_{str(chosen_appendix)}.npy", allow_pickle=True)
            else:
                if args.goal_shape_name in small_list:
                    tool_seq = np.zeros([act_seq.shape[0], 1, 1])
                else:
                    tool_seq = np.ones([act_seq.shape[0], 1, 1])
        except:
            print('opt not found')
            init_pose_seq = np.load(f"{control_out_dir}/init_pose_seq_{str(2)}.npy", allow_pickle=True)
            act_seq = np.load(f"{control_out_dir}/act_seq_{str(2)}.npy", allow_pickle=True)
            if os.path.exists(f"{control_out_dir}/tool_seq_{str(2)}.npy"):
                tool_seq = np.load(f"{control_out_dir}/tool_seq_{str(2)}.npy", allow_pickle=True)
            else:
                if args.goal_shape_name in small_list:
                    tool_seq = np.zeros([act_seq.shape[0], 1, 1])
                else:
                    tool_seq = np.ones([act_seq.shape[0], 1, 1])

        files = glob.glob(control_out_dir+'/*_rgb.png')
        for f in files:
            import pdb; pdb.set_trace()
            os.remove(f)

        print(init_pose_seq.shape, act_seq.shape)

        for i in range(act_seq.shape[0]):
            print(f"folder {index}, grip {i}")
            if args.goal_shape_name == 'D' and i == 0:
                import pdb; pdb.set_trace()
                continue
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
                rgb_img, depth_img = env.render(mode='get')
                # import pdb; pdb.set_trace()
                rgb_img = np.flip(rgb_img, 0)
                # rgb_img = np.flip(rgb_img, 1)
                imageio.imwrite(f"{control_out_dir}/{true_idx:03d}_rgb.png", rgb_img)

        os.system(
            f'ffmpeg -y -i {control_out_dir}/%03d_rgb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {control_out_dir}/vid000.mp4')


if __name__ == "__main__":
    main()