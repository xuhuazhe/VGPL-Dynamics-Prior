import os
import numpy as np
import torch
from tqdm import tqdm, trange
import copy
import imageio

import matplotlib.pyplot as plt
import pdb

from config import gen_args
from data_utils import load_data, get_scene_info, get_env_group, prepare_input
from models import Model, EarthMoverLoss
from utils import create_instance_colors, set_seed,  Tee, count_parameters

from plb.engine.taichi_env import TaichiEnv
from plb.config import load

class Planner(object):
    def __init__(self, args, n_his, n_particle, n_shape, scene_params, model, dist_func, use_gpu, 
                n_grips, len_per_grip, len_per_grip_back, n_shapes_per_gripper, n_shapes_floor, 
                beta_filter=1, env="gripper"):
        self.args = args
        self.n_his = n_his
        self.n_particle = n_particle
        self.n_shape = n_shape
        self.scene_params = scene_params
        self.beta_filter = beta_filter
        self.env = env
        self.model = model
        self.use_gpu = use_gpu
        self.reward_weight = 10.0
        if dist_func == 'emd':
            self.dist_func = EarthMoverLoss()

        self.n_grips = n_grips
        self.len_per_grip = len_per_grip
        self.len_per_grip_back = len_per_grip_back
        self.n_shapes_per_gripper = n_shapes_per_gripper
        self.n_shapes_floor = n_shapes_floor

        self.n_sample = args.control_sample_size
        self.batch_size = args.control_batch_size

        self.mid_point = np.array([0.5, 0.4, 0.5, 0, 0, 0])
        self.default_h = 0.14
        self.sample_radius = 0.25
        self.gripper_rate = 0.01
    
    def trajectory_optimization(
        self,
        state_cur,      # [n_his, state_dim]
        state_goal,     # [state_dim]
        # act_seq,        # [n_his + n_look_ahead - 1, action_dim]
        n_look_ahead,
        n_update_iter,
        # action_lower_lim,
        # action_upper_lim,
        # action_lower_delta_lim,
        # action_upper_delta_lim,
        reward_scale_factor=1.0
    ):
            
        n_batch = int(self.n_sample / self.batch_size)

        for i in range(n_update_iter):
            init_pose_seqs_batch = []
            act_seqs_batch = []
            reward_seqs_batch = []
            state_cur_seqs_batch = []
            
            for _, j in enumerate(tqdm(range(n_batch), total=n_batch)):
                # print(f"Batch: {j}/{n_batch}")
                if i > 0 and self.args.opt_algo == 'CEM':
                    init_pose_seqs, act_seqs = self.sample_gaussian(init_pose_seqs_mean, init_pose_seqs_std)
                else:
                    init_pose_seqs, act_seqs = self.sample_action_params()
                
                print(f"Init poses: {init_pose_seqs.shape}")
                print(f"Actions: {act_seqs.shape}")
                
                # act_seqs = self.sample_action_sequences(
                #     act_seq,
                #     action_lower_lim, action_upper_lim,
                #     action_lower_delta_lim, action_upper_delta_lim)

                state_seqs = self.model_rollout(
                    state_cur, init_pose_seqs, act_seqs, n_look_ahead)
                # print(state_seqs)
                reward_seqs = reward_scale_factor * self.evaluate_traj(state_seqs, state_goal)
                print(reward_seqs)
                reward_seqs = reward_seqs.data.cpu().numpy()

                init_pose_seqs_batch.append(init_pose_seqs)
                act_seqs_batch.append(act_seqs)
                reward_seqs_batch.append(reward_seqs)
                state_cur_seqs_batch.append(state_seqs[:, -self.n_his:, :, :])

            init_pose_seqs_batch = np.concatenate(init_pose_seqs_batch, axis=0)
            act_seqs_batch = np.concatenate(act_seqs_batch, axis=0)
            reward_seqs_batch = np.concatenate(reward_seqs_batch, axis=0)
            state_cur_seqs_batch = torch.cat(state_cur_seqs_batch, 0)

            print('update_iter %d/%d, max: %.4f, mean: %.4f, std: %.4f' % (
                i, n_update_iter, np.max(reward_seqs_batch), np.mean(reward_seqs_batch), np.std(reward_seqs_batch)))

            if self.args.opt_algo == 'max':
                init_pose_seq, act_seq, state_cur_seq = self.optimize_action_max(init_pose_seqs_batch, act_seqs_batch, reward_seqs_batch, state_cur_seqs_batch)
            elif self.args.opt_algo == 'CEM':
                init_pose_seqs_mean, init_pose_seqs_std = self.optimize_action_CEM(init_pose_seqs_batch, act_seqs_batch, reward_seqs_batch)
                print(f"Init poses mean: {init_pose_seqs_mean.shape}")
                print(f"Init poses std: {init_pose_seqs_std.shape}")
                init_pose_seq = init_pose_seqs_mean
                act_seq = self.get_action_seq_from_pose(init_pose_seq)
                state_cur_seq = None # placeholder
            elif self.args.opt_algo == 'MPPI':
                init_pose_seq, act_seq = self.optimize_action_MPPI(init_pose_seqs_batch, act_seqs_batch, reward_seqs_batch)
            else:
                raise NotImplementedError

        # act_seq: [n_his + n_look_ahead - 1, action_dim]
        return init_pose_seq, act_seq, state_cur_seq

    def get_pose(self, new_mid_point, rot_noise):
        x1 = new_mid_point[0] - self.sample_radius * np.cos(rot_noise)
        y1 = new_mid_point[2] + self.sample_radius * np.sin(rot_noise)
        x2 = new_mid_point[0] + self.sample_radius * np.cos(rot_noise)
        y2 = new_mid_point[2] - self.sample_radius * np.sin(rot_noise)

        prim1 = [x1, self.default_h, y1, 1, 0, 0, 0] 
        prim2 = [x2, self.default_h, y2, 1, 0, 0, 0]

        new_prim1 = []
        for j in range(self.n_shapes_per_gripper):
            prim1_tmp = np.concatenate(([prim1[0], prim1[1] + 0.018 * (j-5), prim1[2]], prim1[3:]), axis=None)
            new_prim1.append(prim1_tmp)
        new_prim1 = np.stack(new_prim1)
    
        new_prim2 = []
        for j in range(self.n_shapes_per_gripper):
            prim2_tmp = np.concatenate(([prim2[0], prim2[1] + 0.018 * (j-5), prim2[2]], prim2[3:]), axis=None)
            new_prim2.append(prim2_tmp)
        new_prim2 = np.stack(new_prim2)

        init_pose = np.concatenate((new_prim1, new_prim2), axis=1)

        return init_pose

    def get_action_seq(self, rot_noise):
        delta_g = np.random.uniform(0.27, 0.35)
        counter = 0
        actions = []
        
        while delta_g > 0 and counter < self.len_per_grip:
            x = self.gripper_rate * np.cos(rot_noise)
            y = - self.gripper_rate * np.sin(rot_noise)
            delta_g -= 2 * self.gripper_rate
            actions.append(np.concatenate([np.array([x/0.02,0,y/0.02]), np.zeros(3), 
                                            np.array([-x/0.02,0,-y/0.02]), np.zeros(3)]))
            counter += 1

        actions = actions[:self.len_per_grip]

        for _ in range(self.len_per_grip - len(actions)):
            actions.append(np.concatenate([np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]))

        counter = 0
        while counter < self.len_per_grip_back:
            x = - self.gripper_rate * np.cos(rot_noise)
            y = self.gripper_rate * np.sin(rot_noise)
            actions.append(np.concatenate([np.array([x/0.02,0,y/0.02]), np.zeros(3), 
                                            np.array([-x/0.02,0,-y/0.02]), np.zeros(3)]))
            counter += 1

        return actions

    def get_action_seq_from_pose(self, init_pose_seq):
        gripper_mid_pt = int((self.n_shapes_per_gripper - 1) / 2)
        mid_point_x = np.full(self.n_grips, self.mid_point[0])
        
        cos = (init_pose_seq[:, gripper_mid_pt, 0] - mid_point_x) / self.sample_radius
        if cos.any() < -1.0 or cos.any() > 1.0:
            raise ValueError
        print(f"pose: {init_pose_seq[:, gripper_mid_pt, 0]}; cos: {cos}")
        rot_noise_seq = np.arccos(cos)
        print(f"rot_noise_seq: {rot_noise_seq}")

        act_seq = []
        for rot_noise in rot_noise_seq:
            act_seq.append(self.get_action_seq(rot_noise))

        act_seq = np.stack(act_seq)

        return act_seq

    def sample_action_params(self):
        init_pose_seqs = []
        act_seqs = []
        n_sampled = 0
        while n_sampled < self.batch_size:
            init_pose_seq = []
            act_seq = []
            for i in range(self.n_grips):
                p_noise = np.clip(np.array([0, 0, np.random.randn()*0.06]), a_max=0.1, a_min=-0.1)
                new_mid_point = self.mid_point[:3] + p_noise
                rot_noise = np.random.uniform(0, np.pi)
                
                init_pose = self.get_pose(new_mid_point, rot_noise)
                # print(init_pose.shape)
                init_pose_seq.append(init_pose)

                actions = self.get_action_seq(rot_noise)
                # print(actions.shape)
                act_seq.append(actions)

            init_pose_seq = np.stack(init_pose_seq)
            init_pose_seqs.append(init_pose_seq)

            act_seq = np.stack(act_seq)
            act_seqs.append(act_seq)

            n_sampled += 1

        return np.stack(init_pose_seqs), np.stack(act_seqs)

    def sample_gaussian(self, mean, std):
        init_pose_seqs = []
        act_seqs = []
        i = 0
        while i < self.batch_size:
            init_pose_seq = np.random.normal(mean, std)
            try:
                act_seq = self.get_action_seq_from_pose(init_pose_seq)
            except ValueError:
                print("Invalid init pose sample!")
                continue    
            init_pose_seqs.append(init_pose_seq)
            act_seqs.append(act_seq)
            i += 1
        init_pose_seqs = np.stack(init_pose_seqs)
        act_seqs = np.stack(act_seqs)
        return init_pose_seqs, act_seqs

    def sample_action_sequences(
        self,
        init_act_seq,   # [n_his + n_look_ahead - 1, action_dim]
        action_lower_lim,
        action_upper_lim,
        action_lower_delta_lim,
        action_upper_delta_lim,
        noise_type='normal'):
        action_dim = init_act_seq.shape[-1]
        beta_filter = self.beta_filter

        # act_seqs: [n_sample, N, action_dim]
        # act_seqs_delta: [n_sample, N - 1, action_dim]
        act_seqs = np.stack([init_act_seq] * self.n_sample)
        act_seqs_delta = np.stack([init_act_seq[1:] - init_act_seq[:-1]] * self.n_sample)

        # [n_sample, action_dim]
        act_residual = np.zeros([self.n_sample, action_dim])

        # only add noise to future actions
        # init_act_seq[:(n_his - 1)] are past actions
        # The action we are optimizing for the current timestep is act_seq[n_his - 1]

        # actions that go as input to the dynamics network
        for i in range(self.n_his - 2, init_act_seq.shape[0] - 1):
            if noise_type == "normal":
                sigma = 0.01
                noise_sample = np.random.normal(0, sigma, (self.n_sample, 3))
            else:
                raise ValueError("unknown noise type: %s" % (noise_type))

            act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)
            act_seqs_delta[:, i] += act_residual

            # clip delta lim
            act_seqs_delta[:, i] = np.clip(
                act_seqs_delta[:, i], action_lower_delta_lim, action_upper_delta_lim)

            act_seqs[:, i + 1] = act_seqs[:, i] + act_seqs_delta[:, i]

            # clip absolute lim
            act_seqs[:, i + 1] = np.clip(
                act_seqs[:, i + 1], action_lower_lim, action_upper_lim)

        # act_seqs: [n_sample, -1, action_dim]
        # print(act_seqs.shape)
        return act_seqs

    def expand(self, info):
        length = len(info.shape)
        if length == 2:
            info = info.expand([self.batch_size, -1])
        elif length == 3:
            info = info.expand([self.batch_size, -1, -1])
        elif length == 4:
            info = info.expand([self.batch_size, -1, -1, -1])
        return info

    def state_action(self, state_cur, act_cur):
        state = state_cur[1]  # n_sample x n_his x (n_particle + n_shape) x state_dim
        if self.env == "gripper":
            shapes = state[:, :, self.n_particle+1:, :] ### TODO: This is for gripper
        else:
            raise NotImplementedError
        shapes_diff = shapes[:, 1:, :, :] - shapes[:, :-1, :, :]
        shapes_diff_act = act_cur * 0.02
        pdb.set_trace()

    def prepare_rollout(self):
        B = self.batch_size
        self.scene_params = self.scene_params.expand(self.batch_size, -1)
        self.group_gt = get_env_group(self.args, self.n_particle, self.scene_params, use_gpu=self.use_gpu)
        self.memory_init = self.model.init_memory(B, self.n_particle + self.n_shape)

    def expand_inputs(self, inputs):
        inputs_new = []
        for infos in inputs:
            if infos is not None:
                if isinstance(infos, list):
                    my_info = []
                    for info in infos:
                        info = self.expand(info)
                        my_info.append(info)
                    infos = my_info
                else:
                    infos = self.expand(infos)
            inputs_new.append(infos)
        return inputs_new

    def model_rollout(
        self,
        state_cur,      # [1, n_his, state_dim]
        init_pose_seqs_np,
        act_seqs_np,    # [n_sample, -1, action_dim]
        n_look_ahead):

        init_pose_seqs = torch.FloatTensor(init_pose_seqs_np).float()
        act_seqs = torch.FloatTensor(act_seqs_np).float()
        if self.use_gpu:
            act_seqs = act_seqs.cuda()
            init_pose_seqs = init_pose_seqs.cuda()

        # states_cur: [n_sample, n_his, state_dim]

        states_pred_list = []
        # assert n_look_ahead == act_seqs.shape[1] - self.n_his + 1
        if self.use_gpu:
            state_cur = self.expand(state_cur.unsqueeze(0)).cuda()
        else:
            state_cur = self.expand(state_cur.unsqueeze(0))

        # print(state_cur.shape)

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
                for k in range(self.batch_size):
                    # pdb.set_trace()
                    attr, _, Rr_cur, Rs_cur, cluster_onehot = prepare_input(state_cur[k][-1].cpu().numpy(), self.n_particle,
                                                                            self.n_shape, self.args, stdreg=self.args.stdreg)
                    if self.use_gpu:
                        attr = attr.cuda()
                        Rr_cur = Rr_cur.cuda()
                        Rs_cur = Rs_cur.cuda()
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
                    if self.use_gpu:
                        Rr = torch.cat([Rr, torch.zeros((1, max_n_rel - Rr.size(1), self.n_particle + self.n_shape), device='cuda')], 1)
                        Rs = torch.cat([Rs, torch.zeros((1, max_n_rel - Rs.size(1), self.n_particle + self.n_shape), device='cuda')], 1)
                    else:
                        Rr = torch.cat([Rr, torch.zeros(1, max_n_rel - Rr.size(1), self.n_particle + self.n_shape)], 1)
                        Rs = torch.cat([Rs, torch.zeros(1, max_n_rel - Rs.size(1), self.n_particle + self.n_shape)], 1)
                    Rr_curs[k], Rs_curs[k] = Rr, Rs

                Rr_curs = torch.cat(Rr_curs, dim=0)
                Rs_curs = torch.cat(Rs_curs, dim=0)

                inputs = [attrs, state_cur, Rr_curs, Rs_curs, self.memory_init, self.group_gt, None]
                # inputs = self.expand_inputs(inputs)

                # act_cur = act_seqs[:, i:i + self.n_his]
                # state_cur_act = self.state_action(state_new, act_cur)
                # states_pred: [n_sample, state_dim]
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
        if self.use_gpu:
            state_goal = state_goal.cuda()

        # print(state_seqs.shape, state_goal.shape)
        reward_seqs = []
        for i in range(state_seqs.shape[0]):
            # smaller loss, larger reward
            emd_loss = self.dist_func(state_seqs[i, -1].unsqueeze(0), state_goal)
            reward_seqs.append(0.0 - emd_loss)
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

        idx = np.argmax(reward_seqs)
        print(f"Selected idx: {idx} with loss {reward_seqs[idx]}")
        # [-1, action_dim]
        return init_pose_seqs[idx], act_seqs[idx], state_cur_seqs[idx]

    def optimize_action_CEM(    # Cross Entropy Method (CEM)
        self,
        init_pose_seqs,
        act_seqs,       # [n_sample, -1, action_dim]
        reward_seqs,    # [n_sample]
    ):
        idx = np.argsort(reward_seqs)
        top_seqs = reward_seqs[idx[-5:]]
        print(f"Top seqs: {top_seqs}")
        init_pose_seqs_mean = np.mean(init_pose_seqs[idx[-5:]], axis=0)
        init_pose_seqs_std = np.std(init_pose_seqs[idx[-5:]], axis=0)

        return init_pose_seqs_mean, init_pose_seqs_std

    def optimize_action_MPPI(   # Model-Predictive Path Integral (MPPI)
        self,
        init_pose_seqs,
        act_seqs,       # [n_sample, -1, action_dim]
        reward_seqs     # [n_sample]
    ):
        print(f"reward_seqs: {reward_seqs}")
        # [n_sample, 1, 1]
        # reward_seqs_exp = np.exp(self.reward_weight * (reward_seqs - np.mean(reward_seqs)))
        reward_seqs = (reward_seqs - np.mean(reward_seqs)) / np.var(reward_seqs)
        reward_seqs_norm = reward_seqs / np.linalg.norm(reward_seqs)
        reward_seqs_exp = np.exp(self.reward_weight * reward_seqs_norm)
        print(f"reward_seqs_exp: {reward_seqs_exp}")

        # [-1, action_dim]
        eps = 1e-8
        mid_point_x = np.full((self.n_sample, self.n_grips), self.mid_point[0])
        # mid_point_y = np.repeat(self.mid_point[2], [self.batch_size, self.n_grips])
        gripper_mid_pt = int((self.n_shapes_per_gripper - 1) / 2)
        rot_noise_seqs = np.arccos((init_pose_seqs[:, :, gripper_mid_pt, 0] - mid_point_x) / self.sample_radius)
        print(rot_noise_seqs)
        print(reward_seqs_exp.reshape(-1, 1))
        print(reward_seqs_exp.reshape(-1, 1) * rot_noise_seqs)

        rot_noise_seq = np.sum(reward_seqs_exp.reshape(-1, 1) * rot_noise_seqs, axis=0) / (np.sum(reward_seqs_exp) + eps)
        # act_seq = np.sum(reward_seqs_exp.reshape(-1, 1, 1, 1) * act_seqs, axis=0) / (np.sum(reward_seqs_exp) + eps)

        print(f"rot_noise_seq: {rot_noise_seq}")

        init_pose_seq = []
        act_seq = []
        for rot_noise in rot_noise_seq:
            init_pose_seq.append(self.get_pose(self.mid_point, rot_noise))
            act_seq.append(self.get_action_seq(rot_noise))

        init_pose_seq = np.stack(init_pose_seq)
        act_seq = np.stack(act_seq)
        
        # [-1, action_dim]
        return init_pose_seq, act_seq

def set_parameters(env: TaichiEnv, yield_stress, E, nu):
        env.simulator.yield_stress.fill(yield_stress)
        _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        env.simulator.mu.fill(_mu)
        env.simulator.lam.fill(_lam)

def set_action_limit(all_actions, ctrl_init_idx):
    action_lower_lim = np.min(np.array(all_actions[ctrl_init_idx - 1:]), 0)
    action_upper_lim = np.max(np.array(all_actions[ctrl_init_idx - 1:]), 0)
    action_lim_range = action_upper_lim - action_lower_lim
    action_lower_lim -= action_lim_range * 0.1
    action_upper_lim += action_lim_range * 0.1
    print('action_lower_lim', action_lower_lim)
    print('action_upper_lim', action_upper_lim)
    action_lower_delta_lim = np.min(np.array(all_actions[ctrl_init_idx:]) - np.array(all_actions[ctrl_init_idx - 1:-1]),
                                    0)
    action_upper_delta_lim = np.max(np.array(all_actions[ctrl_init_idx:]) - np.array(all_actions[ctrl_init_idx - 1:-1]),
                                    0)
    action_delta_lim_range = action_upper_delta_lim - action_lower_delta_lim
    action_lower_delta_lim -= action_delta_lim_range * 0.1
    action_upper_delta_lim += action_delta_lim_range * 0.1
    print('action_lower_delta_lim', action_lower_delta_lim)
    print('action_upper_delta_lim', action_upper_delta_lim)

    return action_lower_lim, action_upper_lim, action_lower_delta_lim, action_upper_delta_lim

@profile
def main():
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_control) > 0:
        args.outf = args.outf_control

    control_out_dir = os.path.join(args.outf, 'control')
    os.system('mkdir -p ' + control_out_dir)

    tee = Tee(os.path.join(args.outf, 'control', 'control.log'), 'w')

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
    env.renderer.camera_pos[2] = 2.2
    env.renderer.camera_rot = (0.8, 0.0)

    env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
    env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])
    
    set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2

    # env.render('plt')
    
    use_gpu = True
    task_name = 'gripper'
    n_look_ahead = 20
    action_dim = taichi_env.primitives.action_dim
    action = np.zeros([n_look_ahead, action_dim])
    i = 0
    count = i / 50
    # map count from 0...1 to -1...1
    count = 2 * count - 1
    updown = count * 0.12
    grip_motion = np.random.uniform(0.20, 0.24)
    action[:20, 2] = updown #0.1
    action[:20, 1] = -0.7
    action[:20, 8] = updown #0.1
    action[:20, 7] = -0.7

    for idx, act in enumerate(tqdm(action, total=n_look_ahead)):

        obs = env.step(act)
        if task_name == 'gripper':
            primitive_state = [env.primitives.primitives[0].get_state(0), env.primitives.primitives[1].get_state(0)]
        else:
            primitive_state = [env.primitives.primitives[0].get_state(0)]

        # if (idx+1) % 5 == 0:
        #     env.render(mode='plt')

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
    if use_gpu:
        model = model.cuda()


    # load data (state, actions)
    rollout_dir = f"./data/data_ngrip_new/train/"
    n_vid = 1
    data_names = ['positions', 'shape_quats', 'scene_params']

    n_grips = 3
    len_per_grip = 20
    len_per_grip_back = 10
    n_shapes_floor = 9
    n_shapes_per_gripper = 11

    unit_quat_pad = np.tile([1, 0, 0, 0], (n_shapes_per_gripper, 1))
    ctrl_init_idx = args.n_his
    n_update_delta = 1
    n_update_iter_init = 1
    n_update_iter = 5
    dist_func = 'emd'

    for i in range(n_vid):
        B = 1
        n_particle, n_shape = 0, 0
        init_pose_gt = []
        all_p = []
        all_s = []
        act_seq_gt = []
        actions = []
        for t in range(args.time_step):
            if task_name == "gripper":
                frame_name = str(t) + '.h5'
                if args.shape_aug:
                    frame_name = 'shape_' + frame_name
                frame_path = os.path.join(rollout_dir, str(i).zfill(3), frame_name)
            else:
                raise NotImplementedError
            this_data = load_data(data_names, frame_path)
            if n_particle == 0 and n_shape == 0:
                n_particle, n_shape, scene_params = get_scene_info(this_data)
                scene_params = torch.FloatTensor(scene_params).unsqueeze(0)
                g1_idx = n_particle + n_shapes_floor
                g2_idx = g1_idx + n_shapes_per_gripper

            states = this_data[0]
            if t % (len_per_grip + len_per_grip_back) == 0:
                init_pose_gt.append(np.concatenate((states[g1_idx: g2_idx], unit_quat_pad, states[g2_idx:], unit_quat_pad), axis=1))
            
            if t >= 1:
                all_p.append(states)
                all_s.append(this_data[1])

                action = np.concatenate([(states[g1_idx] - prev_states[g1_idx]) / 0.02, np.zeros(3),
                                        (states[g2_idx] - prev_states[g2_idx]) / 0.02, np.zeros(3)])
                
                actions.append(action)
            prev_states = states

            if len(actions) == len_per_grip + len_per_grip_back:
                act_seq_gt.append(actions)
                # print(len(actions))
                actions = []

            if t == args.time_step - 1:
                while len(actions) < len_per_grip + len_per_grip_back:
                    actions.append(actions[-1])
                # print(len(actions))
                act_seq_gt.append(actions)

    # actions = act_seq_gt[:args.n_his - 1]
    # # duplicate the last action #n_look_ahead times
    # for i in range(n_look_ahead):
    #     actions.append(actions[-1])
    # actions = np.array(actions)

    # action_lower_lim, action_upper_lim, action_lower_delta_lim, action_upper_delta_lim = \
    #     set_action_limit(all_actions=act_seq_gt, ctrl_init_idx=ctrl_init_idx)

    st_idx = ctrl_init_idx
    ed_idx = ctrl_init_idx + 1 # + n_look_ahead

    init_pose_gt = np.expand_dims(init_pose_gt, axis=0)
    act_seq_gt = np.expand_dims(act_seq_gt, axis=0)
    print(f"GT: init pose: {init_pose_gt.shape}; actions: {act_seq_gt.shape}")

    # p_list = copy.copy(all_p[:args.n_his])
    # s_list = copy.copy(all_s[:args.n_his])
    state_goal = torch.FloatTensor(all_p[-1]).unsqueeze(0)[:, :n_particle, :]

    planner = Planner(args=args, n_his=args.n_his, n_particle=n_particle, n_shape=n_shape, scene_params=scene_params,
                    model=model, dist_func=dist_func, use_gpu=use_gpu, n_grips=n_grips,
                    len_per_grip=len_per_grip, len_per_grip_back=len_per_grip_back, 
                    n_shapes_per_gripper=n_shapes_per_gripper, n_shapes_floor=n_shapes_floor)
    planner.prepare_rollout()
    
    ### We now have n_his states, n_his - 1 actions
    for i in range(st_idx, ed_idx):
        print("\n### Step %d/%d" % (i, ed_idx))
        
        if i == st_idx or i % n_update_delta == 0:
            # update the action sequence every n_update_delta iterations
            with torch.set_grad_enabled(False):
                state_cur = torch.FloatTensor(np.stack(all_p[:args.n_his]))
                # print(state_cur.shape)
                # state_seqs = planner.model_rollout(state_cur, init_pose_gt, act_seq_gt, n_look_ahead, use_gpu)
                # reward_seqs = planner.evaluate_traj(state_seqs, state_goal)
                # print(f"GT reward: {reward_seqs}")
    
                # # init_pose_seq, act_seq = planner.trajectory_optimization(
                #     state_cur=state_cur,
                #     state_goal=state_goal,
                #     # act_seq=actions[i-args.n_his:],
                #     n_look_ahead=n_look_ahead - (i - ctrl_init_idx),
                #     n_update_iter=n_update_iter_init if i == st_idx else n_update_iter,
                #     # action_lower_lim=action_lower_lim,
                #     # action_upper_lim=action_upper_lim,
                #     # action_lower_delta_lim=action_lower_delta_lim,
                #     # action_upper_delta_lim=action_upper_delta_lim
                # )

                init_pose_seq, act_seq, _ = planner.trajectory_optimization(
                    state_cur=state_cur,
                    state_goal=state_goal,
                    # act_seq=actions[i-args.n_his:],
                    n_look_ahead=n_look_ahead - (i - ctrl_init_idx),
                    n_update_iter=n_update_iter,
                    # action_lower_lim=action_lower_lim,
                    # action_upper_lim=action_upper_lim,
                    # action_lower_delta_lim=action_lower_delta_lim,
                    # action_upper_delta_lim=action_upper_delta_lim
                )

                # init_pose_seq = []
                # act_seq = []
                # for i in range(n_grips):
                #     if i > 0:
                #         start_idx = i * (len_per_grip + len_per_grip_back)
                #         state_cur_gt = torch.FloatTensor(np.stack(all_p[start_idx:start_idx+args.n_his]))
                #         if use_gpu: state_cur_gt = state_cur_gt.cuda()
                #         state_cur = torch.cat((state_cur, state_cur_gt[:, n_particle:, :]), 1)
                    
                #     # print(state_cur.shape)
                    
                #     # start_idx = i * (len_per_grip + len_per_grip_back)
                #     # state_cur = torch.FloatTensor(np.stack(all_p[start_idx:start_idx+args.n_his]))
                #     end_idx = min((i + 1) * (len_per_grip + len_per_grip_back) - 1, len(all_p) - 1)
                #     state_goal = torch.FloatTensor(all_p[end_idx]).unsqueeze(0)[:, :n_particle, :]

                #     init_pose, actions, state_cur = planner.trajectory_optimization(
                #         state_cur=state_cur,
                #         state_goal=state_goal,
                #         # act_seq=actions[i-args.n_his:],
                #         n_look_ahead=n_look_ahead - (i - ctrl_init_idx),
                #         n_update_iter=n_update_iter_init if i == st_idx else n_update_iter,
                #         # action_lower_lim=action_lower_lim,
                #         # action_upper_lim=action_upper_lim,
                #         # action_lower_delta_lim=action_lower_delta_lim,
                #         # action_upper_delta_lim=action_upper_delta_lim
                #     )
                #     # print(init_pose.shape, actions.shape)
                #     init_pose_seq.append(init_pose)
                #     act_seq.append(actions)
                
                # init_pose_seq = np.concatenate(init_pose_seq, axis=0)
                # act_seq = np.concatenate(act_seq, axis=0)

                print(init_pose_seq.shape, act_seq.shape)


    # env_act = np.zeros([action.shape[0], 12])
    # env_act[:, :3] = action
    # env_act[:, 6:9] = action * np.array([-1, 1, 1])

    os.system('mkdir -p ' + f"{control_out_dir}/000")

    # init_pose_seq = init_pose_gt[0]
    # act_seq = act_seq_gt[0]

    gripper_mid_pt = int((n_shapes_per_gripper - 1) / 2)
    for i in range(act_seq.shape[0]):
        env.primitives.primitives[0].set_state(0, init_pose_seq[i, gripper_mid_pt, :7])
        env.primitives.primitives[1].set_state(0, init_pose_seq[i, gripper_mid_pt, 7:])
        for j in range(act_seq.shape[1]):
            true_idx = i * act_seq.shape[1] + j
            env.step(act_seq[i][j])
            rgb_img, depth_img = env.render(mode='get')
            imageio.imwrite(f"{control_out_dir}/000/{true_idx:03d}_rgb.png", rgb_img)

    os.system(f'ffmpeg -y -i {control_out_dir}/000/%03d_rgb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {control_out_dir}/000/vid000.mp4')         


if __name__ == '__main__':
    main()
