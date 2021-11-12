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

from sys import platform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_params = {
    "mid_point": np.array([0.5, 0.4, 0.5, 0, 0, 0]),
    "default_h": 0.14,
    "sample_radius": 0.25,
    "n_grips": 3,
    "gripper_rate": 0.0075,
    "len_per_grip": 20,
    "len_per_grip_back": 10,
    "n_shapes_floor": 9,
    "n_shapes_per_gripper": 11,
    "gripper_mid_pt": int((11 - 1) / 2)
}


def get_pose(new_mid_point, rot_noise):
    if not torch.is_tensor(new_mid_point):
        new_mid_point = torch.tensor(new_mid_point).to(device)
    
    if not torch.is_tensor(rot_noise):
        rot_noise = torch.tensor(rot_noise).to(device)

    x1 = new_mid_point[0] - task_params["sample_radius"] * torch.cos(rot_noise)
    y1 = new_mid_point[2] + task_params["sample_radius"] * torch.sin(rot_noise)
    x2 = new_mid_point[0] + task_params["sample_radius"] * torch.cos(rot_noise)
    y2 = new_mid_point[2] - task_params["sample_radius"] * torch.sin(rot_noise)

    unit_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(device)

    # import pdb; pdb.set_trace()
    new_prim1 = []
    for j in range(task_params["n_shapes_per_gripper"]):
        prim1_pos = torch.stack([x1, torch.tensor(task_params["default_h"] + 0.018 * (j-5)).to(device), y1])
        prim1_tmp = torch.cat((prim1_pos, unit_quat))
        new_prim1.append(prim1_tmp)
    new_prim1 = torch.stack(new_prim1)

    new_prim2 = []
    for j in range(task_params["n_shapes_per_gripper"]):
        prim2_pos = torch.stack([x2, torch.tensor(task_params["default_h"] + 0.018 * (j-5)).to(device), y2])
        prim2_tmp = torch.cat((prim2_pos, unit_quat))
        new_prim2.append(prim2_tmp)
    new_prim2 = torch.stack(new_prim2)

    init_pose = torch.cat((new_prim1, new_prim2), 1)

    return init_pose


def get_action_seq(rot_noise, gripper_rate):
    # import pdb; pdb.set_trace()
    if not torch.is_tensor(rot_noise):
        rot_noise = torch.tensor(rot_noise).to(device)
    
    if not torch.is_tensor(gripper_rate):
        gripper_rate = torch.tensor(gripper_rate).to(device)

    zero_pad = torch.zeros(3).to(device)
    actions = []
    counter = 0
    while counter < task_params["len_per_grip"]:
        x = gripper_rate * torch.cos(rot_noise)
        y = -gripper_rate * torch.sin(rot_noise)
        prim1_act = torch.stack([x/0.02, torch.tensor(0).to(device), y/0.02])
        prim2_act = torch.stack([-x/0.02, torch.tensor(0).to(device), -y/0.02])
        act = torch.cat((prim1_act, zero_pad, prim2_act, zero_pad))
        actions.append(act)
        counter += 1

    counter = 0
    while counter < task_params["len_per_grip_back"]:
        x = -gripper_rate * torch.cos(rot_noise)
        y = gripper_rate * torch.sin(rot_noise)
        prim1_act = torch.stack([x/0.02, torch.tensor(0).to(device), y/0.02])
        prim2_act = torch.stack([-x/0.02, torch.tensor(0).to(device), -y/0.02])
        act = torch.cat((prim1_act, zero_pad, prim2_act, zero_pad))
        actions.append(act)
        counter += 1

    actions = torch.stack(actions)

    return actions


def get_center_and_rot_from_pose(init_pose_seq):
    # import pdb; pdb.set_trace()
    if not torch.is_tensor(init_pose_seq):
        init_pose_seq = torch.tensor(init_pose_seq).to(device)
        
    mid_point_seq = (init_pose_seq[:, task_params["gripper_mid_pt"], :3] + init_pose_seq[:, task_params["gripper_mid_pt"], 7:10]) / 2

    angle_seq = torch.atan2(init_pose_seq[:, task_params["gripper_mid_pt"], 2] - mid_point_seq[:, 2], \
        init_pose_seq[:, task_params["gripper_mid_pt"], 0] - mid_point_seq[:, 0])

    pi = torch.full(angle_seq.shape, math.pi).to(device)
    angle_seq_new = pi - angle_seq
    
    return mid_point_seq, angle_seq_new


def get_action_seq_from_pose(init_pose_seq, gripper_rate):
    # import pdb; pdb.set_trace()
    _, rot_noise_seq = get_center_and_rot_from_pose(init_pose_seq)
    act_seq = []
    for rot_noise in rot_noise_seq:
        act_seq.append(get_action_seq(rot_noise, gripper_rate))

    act_seq = torch.stack(act_seq)

    return act_seq


def expand(batch_size, info):
    length = len(info.shape)
    if length == 2:
        info = info.expand([batch_size, -1])
    elif length == 3:
        info = info.expand([batch_size, -1, -1])
    elif length == 4:
        info = info.expand([batch_size, -1, -1, -1])
    return info


def visualize_sampled_init_pos(init_pose_seqs, reward_seqs, idx, path):
    init_pose_seqs = init_pose_seqs.cpu().numpy()
    reward_seqs = reward_seqs.cpu().numpy()
    idx = idx.cpu().numpy()

    n_subplots = init_pose_seqs.shape[1]
    fig, axs = plt.subplots(1, n_subplots)
    fig.set_size_inches(8 * n_subplots, 4.8)
    for i in range(n_subplots):
        if n_subplots == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.scatter(init_pose_seqs[:, i, task_params["gripper_mid_pt"], 0], 
            init_pose_seqs[:, i, task_params["gripper_mid_pt"], 2],
            c=reward_seqs, cmap=cm.jet)

        ax.set_title(f"GRIP {i+1}")
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('z coordinate')

    color_map = cm.ScalarMappable(cmap=cm.jet)
    color_map.set_array(reward_seqs[idx])
    plt.colorbar(color_map, ax=axs)

    plt.savefig(path)
    # plt.show()


def visualize_loss(loss_list, path):
    iters, loss = map(list, zip(*loss_list))
    plt.figure(figsize=[16, 9])
    plt.plot(iters, loss, linewidth=6, label='EMD')
    plt.xlabel('epochs', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Test Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.savefig(path)
    # plt.show()


class Planner(object):
    def __init__(self, args, taichi_env, env_init_state, scene_params, n_particle, n_shape, model, all_p, 
                task_params, use_gpu, rollout_path, env="gripper"):
        self.args = args
        self.taichi_env = taichi_env
        self.env_init_state = env_init_state
        self.scene_params = scene_params
        self.n_particle = n_particle
        self.n_shape = n_shape
        self.model = model
        self.all_p = all_p
        self.task_params = task_params
        self.use_gpu = use_gpu
        self.rollout_path = rollout_path
        self.env = env

        self.n_his = args.n_his
        self.use_sim = args.use_sim
        self.dist_func = args.rewardtype
        self.n_sample = args.control_sample_size
        self.batch_size = args.control_batch_size
        self.sample_iter = args.sample_iter
        self.sample_iter_cur = 0
        self.opt_iter = args.opt_iter
        self.opt_iter_cur = 0

        self.mid_point = task_params["mid_point"]
        self.default_h = task_params["default_h"]
        self.sample_radius = task_params["sample_radius"]
        self.n_grips = task_params["n_grips"]
        self.gripper_rate = task_params["gripper_rate"]
        self.len_per_grip = task_params["len_per_grip"]
        self.len_per_grip_back = task_params["len_per_grip_back"]
        self.n_shapes_per_gripper = task_params["n_shapes_per_gripper"]
        self.n_shapes_floor = task_params["n_shapes_floor"]
        self.gripper_mid_pt = task_params["gripper_mid_pt"]

        if args.debug:
            self.n_sample = 8
            self.init_pose_sample_size = 4
            self.gripper_rate_sample_size = 4
            self.n_epochs_GD = 10
            self.best_k_GD = 1
        else:
            self.init_pose_sample_size = 8
            self.gripper_rate_sample_size = 4
            self.n_epochs_GD = 100
            self.best_k_GD = 4


    @profile
    def trajectory_optimization(self):
        init_pose_seq = []
        act_seq = []
        state_cur = None
        for i in range(self.sample_iter):
            self.sample_iter_cur = i

            n_grips_per_iter = int(self.n_grips / self.sample_iter)
            start_idx = i * n_grips_per_iter * (self.len_per_grip + self.len_per_grip_back)
            state_cur_gt = torch.FloatTensor(np.stack(self.all_p[start_idx:start_idx+self.args.n_his]))

            end_idx = min((i + 1) *  n_grips_per_iter * (self.len_per_grip + self.len_per_grip_back) - 1, len(self.all_p) - 1)
            state_goal = torch.FloatTensor(self.all_p[end_idx]).unsqueeze(0)[:, :self.n_particle, :]

            if state_cur == None:
                state_cur = state_cur_gt
            else:
                state_cur = state_cur.to(device) 
                state_cur_gt = state_cur_gt.to(device)

                state_cur = torch.cat((state_cur_seq_opt, state_cur_gt[:, self.n_particle:, :]), 1)
        
            print(f"state_cur: {state_cur.shape}, state_goal: {state_goal.shape}")

            init_pose_seqs_pool, act_seqs_pool = self.sample_action_params(n_grips_per_iter)
            reward_seqs, state_cur_seqs = self.rollout(init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal)

            print('sampling: max: %.4f, mean: %.4f, std: %.4f' % (torch.max(reward_seqs), torch.mean(reward_seqs), torch.std(reward_seqs)))

            if self.args.opt_algo == 'max':
                init_pose_seq_opt, act_seq_opt, state_cur_seq_opt = self.optimize_action_max(init_pose_seqs_pool, act_seqs_pool, reward_seqs, state_cur_seqs)
            elif self.args.opt_algo == 'CEM':
                for i in range(self.opt_iter):
                    self.opt_iter_cur = i
                    if i == self.opt_iter - 1:
                        init_pose_seq_opt, act_seq_opt, state_cur_seq_opt = self.optimize_action_max(init_pose_seqs_pool, act_seqs_pool, reward_seqs, state_cur_seqs)
                    else:
                        init_pose_seqs_pool, act_seqs_pool = self.optimize_action_CEM(init_pose_seqs_pool, act_seqs_pool, reward_seqs)
                        reward_seqs, state_cur_seqs = self.rollout(init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal)
            elif self.args.opt_algo == "GD":
                with torch.set_grad_enabled(True):
                    init_pose_seq_opt, act_seq_opt, state_cur_seq_opt = self.optimize_action_GD(init_pose_seqs_pool, act_seqs_pool, reward_seqs, state_cur, state_goal)
            elif self.args.opt_algo == "CEM_GD":
                for i in range(self.opt_iter):
                    self.opt_iter_cur = i
                    if i == self.opt_iter - 1:
                        with torch.set_grad_enabled(True):
                            init_pose_seq_opt, act_seq_opt, state_cur_seq_opt = self.optimize_action_GD(init_pose_seqs_pool, act_seqs_pool, reward_seqs, state_cur, state_goal)
                    else:
                        init_pose_seqs_pool, act_seqs_pool = self.optimize_action_CEM(init_pose_seqs_pool, act_seqs_pool, reward_seqs)
                        reward_seqs, state_cur_seqs = self.rollout(init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal)
            else:
                raise NotImplementedError

            # print(init_pose.shape, actions.shape)
            init_pose_seq.append(init_pose_seq_opt.cpu().numpy())
            act_seq.append(act_seq_opt.cpu().numpy())
        
        init_pose_seq = np.concatenate(init_pose_seq, axis=0)
        act_seq = np.concatenate(act_seq, axis=0)

        return init_pose_seq, act_seq


    def rollout(self, init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal):
        # import pdb; pdb.set_trace()
        reward_seqs_rollout = []
        state_cur_seqs_rollout = []
        
        n_batch = int(math.ceil(init_pose_seqs_pool.shape[0] / self.batch_size))
        for i, _ in enumerate(tqdm(range(n_batch), total=n_batch)):
            # print(f"Batch: {i}/{n_batch}")
            init_pose_seqs = init_pose_seqs_pool[i*self.batch_size:(i+1)*self.batch_size]
            act_seqs = act_seqs_pool[i*self.batch_size:(i+1)*self.batch_size]

            if self.use_sim:
                state_seqs = self.sim_rollout(init_pose_seqs, act_seqs)
            else:
                # import pdb; pdb.set_trace()
                state_seqs = self.model_rollout(state_cur, init_pose_seqs, act_seqs)
            reward_seqs = self.evaluate_traj(state_seqs, state_goal)
            print(f"reward seqs: {reward_seqs}")
            # reward_seqs = reward_seqs.data.cpu().numpy()

            reward_seqs_rollout.append(reward_seqs)
            state_cur_seqs_rollout.append(state_seqs[:, -self.n_his:, :, :])

        # import pdb; pdb.set_trace()
        reward_seqs_rollout = torch.cat(reward_seqs_rollout, 0)
        state_cur_seqs_rollout = torch.cat(state_cur_seqs_rollout, 0)

        return reward_seqs_rollout, state_cur_seqs_rollout


    def sample_action_params(self, n_grips):
        init_pose_seqs = []
        act_seqs = []
        n_sampled = 0
        while n_sampled < self.n_sample:
            init_pose_seq = []
            act_seq = []
            for i in range(n_grips):
                p_noise = np.clip(np.array([0, 0, np.random.randn()*0.06]), a_max=0.1, a_min=-0.1)
                new_mid_point = self.mid_point[:3] + p_noise
                rot_noise = np.random.uniform(0, np.pi)

                init_pose = get_pose(new_mid_point, rot_noise)
                # print(init_pose.shape)
                init_pose_seq.append(init_pose)

                rate_noise = np.clip(np.random.randn()*0.001, a_max=0.001, a_min=-0.001)
                gripper_rate = task_params["gripper_rate"] + rate_noise
                # print(f'gripper_rate: {gripper_rate}')
                actions = get_action_seq(rot_noise, gripper_rate)
                # print(actions.shape)
                act_seq.append(actions)

            init_pose_seq = torch.stack(init_pose_seq)
            init_pose_seqs.append(init_pose_seq)

            act_seq = torch.stack(act_seq)
            act_seqs.append(act_seq)

            n_sampled += 1

        return torch.stack(init_pose_seqs), torch.stack(act_seqs)


    @profile
    def sim_rollout(self, init_pose_seqs, act_seqs):
        state_seq_batch = []
        for t in range(act_seqs.shape[0]):
            self.taichi_env.set_state(**self.env_init_state)
            state_seq = []
            for i in range(act_seqs.shape[1]):
                self.taichi_env.primitives.primitives[0].set_state(0, init_pose_seqs[t, i, self.gripper_mid_pt, :7])
                self.taichi_env.primitives.primitives[1].set_state(0, init_pose_seqs[t, i, self.gripper_mid_pt, 7:])
                for j in range(act_seqs.shape[2]):
                    self.taichi_env.step(act_seqs[t][i][j])
                    x = self.taichi_env.simulator.get_x(0)
                    step_size = len(x) // self.n_particle
                    # print(f"x before: {x.shape}")
                    x = x[::step_size]
                    particles = x[:300]
                    # print(f"x after: {x.shape}")
                    state_seq.append(particles)

            state_seq = np.stack(state_seq)
            # print(f"rollout state_seq: {state_seq.shape}")
            state_seq_batch.append(state_seq)

        state_seq_batch = torch.from_numpy(np.stack(state_seq_batch)).to(device)

        # print(f"state_seq_batch: {state_seq_batch.shape}")

        return state_seq_batch


    @profile
    def model_rollout(
        self,
        state_cur,      # [1, n_his, state_dim]
        init_pose_seqs,
        act_seqs,    # [n_sample, -1, action_dim]
    ):
        if not torch.is_tensor(init_pose_seqs):
            init_pose_seqs = torch.tensor(init_pose_seqs).to(device)

        if not torch.is_tensor(act_seqs):
            act_seqs = torch.tensor(act_seqs).to(device)

        init_pose_seqs = init_pose_seqs.float()
        act_seqs = act_seqs.float()
        state_cur = expand(init_pose_seqs.shape[0], state_cur.unsqueeze(0)).to(device)

        scene_params = self.scene_params.expand(init_pose_seqs.shape[0], -1)
        group_gt = get_env_group(self.args, self.n_particle, scene_params, use_gpu=self.use_gpu)
        memory_init = self.model.init_memory(init_pose_seqs.shape[0], self.n_particle + self.n_shape)

        # print(state_cur.shape)
        
        # import pdb; pdb.set_trace()
        states_pred_list = []
        # for i in range(min(n_look_ahead, act_seqs.shape[1] - self.n_his + 1)):
        for i in range(act_seqs.shape[1]):
            shape1 = init_pose_seqs[:, i, :, :3]
            shape2 = init_pose_seqs[:, i, :, 7:10]
            for j in range(act_seqs.shape[2]):
                # state_cur = torch.tensor(state_cur_np, device=device).float()
                true_idx = i * (self.len_per_grip + self.len_per_grip_back) + j
                # print(f"{true_idx}/{act_seqs.shape[1] * act_seqs.shape[2]}")
                attrs = []
                Rr_curs = []
                Rs_curs = []
                max_n_rel = 0
                for k in range(act_seqs.shape[0]):
                    # import pdb; pdb.set_trace()
                    attr, _, Rr_cur, Rs_cur, cluster_onehot = prepare_input(state_cur[k][-1].detach().cpu().numpy(), self.n_particle,
                                                                            self.n_shape, self.args, stdreg=self.args.stdreg)
                    attr = attr.to(device)
                    Rr_cur = Rr_cur.to(device)
                    Rs_cur = Rs_cur.to(device)
                    max_n_rel = max(max_n_rel, Rr_cur.size(0))
                    attr = attr.unsqueeze(0)
                    Rr_cur = Rr_cur.unsqueeze(0)
                    Rs_cur = Rs_cur.unsqueeze(0)
                    # state_cur = state_cur.unsqueeze(0)
                    attrs.append(attr)
                    Rr_curs.append(Rr_cur)
                    Rs_curs.append(Rs_cur)

                attrs = torch.cat(attrs, dim=0)
                for k in range(len(Rr_curs)):
                    Rr, Rs = Rr_curs[k], Rs_curs[k]
                    Rr = torch.cat([Rr, torch.zeros((1, max_n_rel - Rr.size(1), self.n_particle + self.n_shape)).to(device)], 1)
                    Rs = torch.cat([Rs, torch.zeros((1, max_n_rel - Rs.size(1), self.n_particle + self.n_shape)).to(device)], 1)
                    Rr_curs[k], Rs_curs[k] = Rr, Rs

                Rr_curs = torch.cat(Rr_curs, dim=0)
                Rs_curs = torch.cat(Rs_curs, dim=0)

                inputs = [attrs, state_cur, Rr_curs, Rs_curs, memory_init, group_gt, None]

                # import pdb; pdb.set_trace()
                pred_pos, pred_motion_norm, std_cluster  = self.model.predict_dynamics(inputs)

                shape1 += act_seqs[:, i, j, :3].unsqueeze(1).expand(-1, self.n_shapes_per_gripper, -1) * 0.02
                shape2 += act_seqs[:, i, j, 6:9].unsqueeze(1).expand(-1, self.n_shapes_per_gripper, -1) * 0.02

                # print(f"pred_pos shape: {pred_pos.shape}\nstate_cur shape: {state_cur.shape}\nshape shape: {shape1.shape}")

                pred_pos = torch.cat([pred_pos, state_cur[:, -1, self.n_particle: self.n_particle + self.n_shapes_floor, :], shape1, shape2], 1)
                # print(f"pred_pos shape: {pred_pos.shape}")

                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
                # print(f"state_cur shape: {state_cur.shape}")

                # print(torch.cuda.memory_summary())
                
                states_pred_list.append(pred_pos[:, :self.n_particle, :])
        
        # states_pred_tensor: [n_sample, n_look_ahead, state_dim]
        states_pred_tensor = torch.stack(states_pred_list, dim=1)

        return states_pred_tensor #.data.cpu().numpy()


    def evaluate_traj(
        self,
        state_seqs,     # [n_sample, n_look_ahead, state_dim]
        state_goal,     # [state_dim]
    ):
        # reward_seqs = -np.mean(np.sum((state_seqs[:, -1] - state_goal)**2, 2), 1)
        # goal = state_goal.expand(self.n_sample, -1, -1)
        state_goal = state_goal.to(device)

        # print(state_seqs.shape, state_goal.shape)
        reward_seqs = []
        for i in range(state_seqs.shape[0]):
            # smaller loss, larger reward
            if self.dist_func == "emd":
                dist_func = EarthMoverLoss()
                loss = dist_func(state_seqs[i, -1].unsqueeze(0), state_goal)
            elif self.dist_func == "l1shape":
                dist_func = L1ShapeLoss()
                loss = dist_func(state_seqs[i, -1], state_goal[0])
            reward_seqs.append(0.0 - loss)
        # reward_seqs: [n_sample]
        reward_seqs = torch.stack(reward_seqs)
        # print(reward_seqs)
        return reward_seqs


    def optimize_action_max(
        self,
        init_pose_seqs,
        act_seqs,       # [n_sample, -1, action_dim]
        reward_seqs,    # [n_sample]
        state_cur_seqs
    ):

        idx = torch.argsort(reward_seqs)
        print(f"Selected idx: {idx[-1]} with loss {reward_seqs[idx[-1]]}")

        visualize_sampled_init_pos(init_pose_seqs, reward_seqs, idx, \
            os.path.join(self.rollout_path, f'plot_max_{self.sample_iter_cur}'))

        # [-1, action_dim]
        return init_pose_seqs[idx[-1]], act_seqs[idx[-1]], state_cur_seqs[idx[-1]]


    def optimize_action_CEM(    # Cross Entropy Method (CEM)
        self,
        init_pose_seqs,
        act_seqs,
        reward_seqs,    # [n_sample]
        best_k_ratio=0.05
    ):
        best_k = max(4, int(init_pose_seqs.shape[0] * best_k_ratio))
        idx = torch.argsort(reward_seqs)
        print(f"Selected top reward seqs: {reward_seqs[idx[-best_k:]]}")
        # print(f"Selected top init pose seqs: {init_pose_seqs[idx[-best_k:], :, self.gripper_mid_pt, :7]}")

        visualize_sampled_init_pos(init_pose_seqs, reward_seqs, idx, \
            os.path.join(self.rollout_path, f'plot_cem_s{self.sample_iter_cur}_o{self.opt_iter_cur}'))

        init_pose_seqs_pool = []
        act_seqs_pool = []
        for i in range(best_k, 0, -1):
            init_pose_seq = init_pose_seqs[idx[-i]]
            # print(f"Selected init pose seq: {init_pose_seq[:, self.gripper_mid_pt, :7]}")
            init_pose_seqs_pool.append(init_pose_seq)
            j = 1

            if i > 1:
                n_init_pose_samples = int(self.init_pose_sample_size / (2**i))
            else:
                n_init_pose_samples = self.init_pose_sample_size - len(init_pose_seqs_pool) + 1
            
            while j < n_init_pose_samples:
                mid_point_seq, angle_seq = get_center_and_rot_from_pose(init_pose_seq)
                init_pose_seq_sample = []
                for k in range(init_pose_seq.shape[0]):
                    p_noise = torch.clamp(torch.tensor([0, 0, np.random.randn()*0.03]), max=0.1, min=-0.1).to(device)
                    rot_noise = torch.clamp(torch.tensor(np.random.randn()*math.pi/36), max=0.1, min=-0.1).to(device)
                
                    new_mid_point = mid_point_seq[k, :3] + p_noise
                    new_angle = angle_seq[k] + rot_noise
                    init_pose = get_pose(new_mid_point, new_angle)
                    init_pose_seq_sample.append(init_pose)

                    # import pdb; pdb.set_trace()

                init_pose_seq_sample = torch.stack(init_pose_seq_sample)
                init_pose_seqs_pool.append(init_pose_seq_sample)
                # print(f"Selected init pose seq: {init_pose_seq_sample[:, self.gripper_mid_pt, :7]}")

                j += 1
        
        for init_pose_seq_sample in init_pose_seqs_pool:
            for i in range(self.gripper_rate_sample_size):
                rate_noise = torch.clamp(torch.tensor(np.random.randn()*0.001), max=0.001, min=-0.001)
                gripper_rate_sample = task_params["gripper_rate"] + rate_noise
                # print(f"{i} gripper_rate_sample: {gripper_rate_sample}")
                act_seq_sample = get_action_seq_from_pose(init_pose_seq_sample, gripper_rate_sample)

                act_seqs_pool.append(act_seq_sample)

        # import pdb; pdb.set_trace()
        init_pose_seqs_pool = torch.stack(init_pose_seqs_pool)
        init_pose_seqs_pool = torch.repeat_interleave(init_pose_seqs_pool, self.gripper_rate_sample_size, dim=0)
        act_seqs_pool = torch.stack(act_seqs_pool)
        print(f"Init pose seq pool: {init_pose_seqs_pool.shape}; Act seq pool: {act_seqs_pool.shape}")

        return init_pose_seqs_pool, act_seqs_pool

    def optimize_action_GD(
        self,
        init_pose_seqs,
        act_seqs,
        reward_seqs,
        state_cur,
        state_goal,
        lr=1e-6
    ):
        state_goal = state_goal.to(device)
        best_k = self.best_k_GD
        idx = torch.argsort(reward_seqs)
        # print(f"Selected idx: {idx[-1]} with loss {reward_seqs[idx[-1]]}")
        print(f"Selected top reward seqs: {reward_seqs[idx[-best_k:]]}")

        best_mid_point_seqs = []
        best_angle_seqs = [] 
        best_gripper_rates = []
        for i in range(best_k, 0, -1):
            best_mid_point_seq, best_angle_seq = get_center_and_rot_from_pose(init_pose_seqs[idx[-i]])
            best_gripper_rate = torch.linalg.norm(act_seqs[idx[-i], :, 0, :3] * 0.02)

            best_mid_point_seqs.append(best_mid_point_seq)
            best_angle_seqs.append(best_angle_seq)
            best_gripper_rates.append(best_gripper_rate)
        
        best_mid_point_seqs = torch.stack(best_mid_point_seqs)
        best_angle_seqs = torch.stack(best_angle_seqs)
        best_gripper_rates = torch.stack(best_gripper_rates)

        mid_points = best_mid_point_seqs.requires_grad_()
        angles = best_angle_seqs.requires_grad_()
        gripper_rates = best_gripper_rates.requires_grad_()

        # import pdb; pdb.set_trace()
        optimizer = torch.optim.Adam([mid_points, angles, gripper_rates], lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

        loss_list = []
        for epoch in range(self.n_epochs_GD):
            init_pose_seq_samples = []
            act_seq_samples = []
            for i in range(mid_points.shape[0]):
                init_pose_seq_sample = []
                for j in range(mid_points.shape[1]):
                    init_pose = get_pose(best_mid_point_seqs[i, j, :3], angles[i, j])
                    init_pose_seq_sample.append(init_pose)

                # import pdb; pdb.set_trace()
                init_pose_seq_sample = torch.stack(init_pose_seq_sample).to(device)
                act_seq_sample = get_action_seq_from_pose(init_pose_seq_sample, gripper_rates[i]).to(device)

                init_pose_seq_samples.append(init_pose_seq_sample)
                act_seq_samples.append(act_seq_sample)

            init_pose_seq_samples = torch.stack(init_pose_seq_samples)
            act_seq_samples = torch.stack(act_seq_samples)

            # import pdb; pdb.set_trace()
            reward_seqs, state_seqs = self.rollout(init_pose_seq_samples, act_seq_samples[:, :, :task_params["len_per_grip"]], state_cur, state_goal)

            loss = torch.min(torch.neg(reward_seqs))

            print(f"Epoch: {epoch}; Loss: {loss.item()}")
            loss_list.append([epoch, loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        visualize_loss(loss_list, os.path.join(self.rollout_path, 'plot_GD_loss'))

        # import pdb; pdb.set_trace()
        idx = torch.argsort(reward_seqs)
        init_pose_seq_opt = []
        for i in range(mid_points[idx[-1]].shape[0]):
            init_pose = get_pose(mid_points[idx[-1], i, :3], angles[idx[-1], i])
            init_pose_seq_opt.append(init_pose)

        init_pose_seq_opt = torch.stack(init_pose_seq_opt).to(device)
        act_seq_opt = get_action_seq_from_pose(init_pose_seq_opt, gripper_rates[idx[-1]])

        print(f"Optimal set of params:\nmid_point: {mid_points[idx[-1]]}\nangle: {angles[idx[-1]]}\ngripper_rate: {gripper_rates[idx[-1]]}")
        print(f"Optimal init pose seq: {init_pose_seq_opt[:, task_params['gripper_mid_pt'], :7]}")

        return init_pose_seq_opt, act_seq_opt, state_seqs[idx[-1]]


@profile
def main():
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_control) > 0:
        args.outf = args.outf_control

    if args.gt_action:
        test_name = f'sim_{args.use_sim}+{args.rewardtype}+gt_action_{args.gt_action}'
    else:
        test_name = f'sim_{args.use_sim}+{args.rewardtype}+sample_iter_{args.sample_iter}+opt_{args.opt_algo}_{args.opt_iter}'

    vid_idx = 0
    control_out_dir = os.path.join(args.outf, 'control', str(vid_idx).zfill(3), test_name)
    os.system('mkdir -p ' + control_out_dir)

    use_gpu = True

    tee = Tee(os.path.join(control_out_dir, 'control.log'), 'w')

    # set up the env
    cfg = load(args.gripperf)
    print(cfg)

    env = None
    state = None
    if platform != 'darwin' and not args.debug:
        env = TaichiEnv(cfg, nn=False, loss=False)
        env.initialize()
        state = env.get_state()

        env.set_state(**state)
        taichi_env = env

        env.renderer.camera_pos[0] = 0.5
        env.renderer.camera_pos[1] = 2.5
        env.renderer.camera_pos[2] = 2.2
        env.renderer.camera_rot = (0.8, 0.0)

        env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
        env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])
        
        def set_parameters(env: TaichiEnv, yield_stress, E, nu):
            env.simulator.yield_stress.fill(yield_stress)
            _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
            env.simulator.mu.fill(_mu)
            env.simulator.lam.fill(_lam)

        set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2


    # load dynamics model
    model = Model(args, use_gpu)
    print("model_kp #params: %d" % count_parameters(model))
    model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter)
    model_path = os.path.join(args.outf, model_name)
    if use_gpu:
        pretrained_dict = torch.load(model_path)
    else:
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    # only load parameters in dynamics_predictor
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if 'dynamics_predictor' in k and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    
    model = model.to(device)


    # load data (state, actions)
    unit_quat_pad = np.tile([1, 0, 0, 0], (task_params["n_shapes_per_gripper"], 1))
    task_name = 'gripper'
    data_names = ['positions', 'shape_quats', 'scene_params']
    rollout_dir = f"./data/data_ngrip_new/train/"
    steps_per_grip = task_params["len_per_grip"] + task_params["len_per_grip_back"]

    init_pose_gt = []
    all_p = []
    all_s = []
    act_seq_gt = []
    actions = []
    for t in range(args.time_step):
        if task_name == "gripper":
            frame_name = str(t) + '.h5'
            if args.gt_state_goal:
                frame_name = 'gt_' + frame_name
            if args.shape_aug:
                frame_name = 'shape_' + frame_name
            frame_path = os.path.join(rollout_dir, str(vid_idx).zfill(3), frame_name)
        else:
            raise NotImplementedError
        this_data = load_data(data_names, frame_path)

        n_particle, n_shape, scene_params = get_scene_info(this_data)
        scene_params = torch.FloatTensor(scene_params).unsqueeze(0)
        g1_idx = n_particle + task_params["n_shapes_floor"]
        g2_idx = g1_idx + task_params["n_shapes_per_gripper"]

        states = this_data[0]
        
        if t >= 1:
            all_p.append(states)
            all_s.append(this_data[1])

            action = np.concatenate([(states[g1_idx] - prev_states[g1_idx]) / 0.02, np.zeros(3),
                                    (states[g2_idx] - prev_states[g2_idx]) / 0.02, np.zeros(3)])
        
            if len(actions) == task_params["len_per_grip"] - 1:
                actions.insert(0, actions[0])
            elif len(actions) == steps_per_grip - 1:
                actions.append(actions[-1])
            else:
                actions.append(action)

            if t == 1: actions.insert(0, actions[0])
            if t == args.time_step - 1: actions.append(actions[-1])

        prev_states = states

        if t % steps_per_grip == 0:
            init_pose_gt.append(np.concatenate((states[g1_idx: g2_idx], unit_quat_pad, states[g2_idx:], unit_quat_pad), axis=1))
        
        if len(actions) == steps_per_grip:
            # print(f"Actions: {actions}")
            act_seq_gt.append(actions)
            # import pdb; pdb.set_trace()
            # hard code
            init_pose_gt[-1] = np.concatenate((init_pose_gt[-1][:, :3] - 2 * 0.02 * np.tile(actions[0][:3], (init_pose_gt[-1].shape[0], 1)), unit_quat_pad, \
                init_pose_gt[-1][:, 7:10] - 2 * 0.02 * np.tile(actions[0][6:9], (init_pose_gt[-1].shape[0], 1)), unit_quat_pad), axis=1)
            actions = []

        prev_states = states

    init_pose_gt = np.expand_dims(init_pose_gt, axis=0)
    act_seq_gt = np.expand_dims(act_seq_gt, axis=0)

    print(f"GT shape: init pose: {init_pose_gt.shape}; actions: {act_seq_gt.shape}")
    print(f"GT init pose: {init_pose_gt[0, :, task_params['gripper_mid_pt'], :7]}")
    # print(act_seq_gt)


    planner = Planner(args=args, taichi_env=env, env_init_state=state, scene_params=scene_params, n_particle=n_particle, 
                    n_shape=n_shape, model=model, all_p=all_p, task_params=task_params, use_gpu=use_gpu, rollout_path=control_out_dir)

    with torch.set_grad_enabled(False):
        if args.gt_action:
            state_cur = torch.FloatTensor(np.stack(all_p[:args.n_his]))
            state_goal = torch.FloatTensor(all_p[-1]).unsqueeze(0)[:, :n_particle, :]
            if args.use_sim:
                state_seqs = planner.sim_rollout(init_pose_gt, act_seq_gt)
            else:
                init_pose_gt_batch = np.repeat(init_pose_gt, args.control_batch_size, axis=0)
                act_seq_gt_batch = np.repeat(act_seq_gt, args.control_batch_size, axis=0)
                state_seqs = planner.model_rollout(state_cur, init_pose_gt_batch, act_seq_gt_batch)
            reward_seqs = planner.evaluate_traj(state_seqs, state_goal)
            print(f"GT reward: {reward_seqs}")
            init_pose_seq = init_pose_gt[0]
            act_seq = act_seq_gt[0]
        else:
            init_pose_seq, act_seq = planner.trajectory_optimization()

    print(init_pose_seq[:, task_params["gripper_mid_pt"], :7])
    print(init_pose_seq.shape, act_seq.shape)

    torch.cuda.empty_cache()

    env.set_state(**state)

    for i in range(act_seq.shape[0]):
        env.primitives.primitives[0].set_state(0, init_pose_seq[i, task_params["gripper_mid_pt"], :7])
        env.primitives.primitives[1].set_state(0, init_pose_seq[i, task_params["gripper_mid_pt"], 7:])
        for j in range(act_seq.shape[1]):
            true_idx = i * act_seq.shape[1] + j
            env.step(act_seq[i][j])
            rgb_img, depth_img = env.render(mode='get')
            imageio.imwrite(f"{control_out_dir}/{true_idx:03d}_rgb.png", rgb_img)

    os.system(f'ffmpeg -y -i {control_out_dir}/%03d_rgb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {control_out_dir}/vid000.mp4')


if __name__ == '__main__':
    main()
