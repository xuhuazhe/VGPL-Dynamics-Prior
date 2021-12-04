import argparse
import copy
import os
import pdb
import time
import cv2

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from config import gen_args
from data_utils import load_data, get_scene_info, normalize_scene_param
from data_utils import get_env_group, prepare_input, denormalize
from models import Model
from utils import add_log, convert_groups_to_colors, train_plot_curves, eval_plot_curves
from utils import create_instance_colors, set_seed, Tee, count_parameters
from models import EarthMoverLoss, ChamferLoss, UpdatedHausdorffLoss

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

import cProfile
import pstats
import io

def visualize_points(ax, all_points, n_particles):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c='b', s=10)
    shapes = ax.scatter(all_points[n_particles:, 0], all_points[n_particles:, 2], all_points[n_particles:, 1], c='r', s=10)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    return points, shapes

def plt_render(particles_set, n_particle, vid_path, render_path):
    p_gt, p_sample, p_pred = particles_set
    particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = p_pred.shape[0]
    rows = 3
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 9))
    row_titles = ['GT', 'Sample', 'Prediction']
    views = [(-90, 90), (0, 0), (-120, 30)]
    plot_info_all = {}
    for i in range(rows):
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        # if i == 0:
        #     plot_info = []
        #     for j in range(cols):
        #         continue
        #         # frame_path = f'{vid_path}/{str(0).zfill(3)}_rgb_{j}.png'
        #         # ax = fig.add_subplot(rows, cols, i * cols + j + 1)
        #         # img = ax.imshow(mpimg.imread(frame_path))
        #         # plot_info.append(img)
        # else:
        states = particles_set[i]
        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    # plt.show()
    plt.tight_layout()

    def update(step):
        outputs = []
        for i in range(rows):
            # if i == 0:
            #     for j in range(cols):
            #         pdb.set_trace()
            #         img = plot_info_all[row_titles[i]][j]
            #         frame_path = f'{vid_path}/{str(step).zfill(3)}_rgb_{j}.png'
            #         img.set_array(mpimg.imread(frame_path))
            #         outputs.append(img)
            # else:
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
    anim.save(render_path, writer=animation.PillowWriter(fps=20))


def evaluate(args, eval_epoch, eval_iter):
    global p1
    global t_step
    global colors
    
    args.evalf = os.path.join(args.outf, 'eval')

    os.system('mkdir -p ' + args.evalf)
    os.system('mkdir -p ' + os.path.join(args.evalf, 'plot'))
    os.system('mkdir -p ' + os.path.join(args.evalf, 'render'))
    # os.system('mkdir -p ' + os.path.join(args.evalf, 'vispy'))

    tee = Tee(os.path.join(args.evalf, 'eval.log'), 'w')

    data_names = args.data_names
    use_gpu = 0 #torch.cuda.is_available()

    # create model and load weights
    model = Model(args, use_gpu)
    print("model_kp #params: %d" % count_parameters(model))

    if eval_epoch < 0:
        model_name = 'net_best.pth'
    else:
        model_name = 'net_epoch_%d_iter_%d.pth' % (eval_epoch, eval_iter)

    model_path = os.path.join(args.outf, model_name)
    print("Loading network from %s" % model_path)

    if args.stage == 'dy':
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu') )
        model_dict = model.state_dict()
        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

    model.eval()

    if use_gpu:
        model = model.cuda()

    emd_loss = EarthMoverLoss()
    chamfer_loss = ChamferLoss()
    uh_loss = UpdatedHausdorffLoss()

    loss_list_over_episodes = []

    for idx_episode in range(0, args.n_rollout, 1): #range(args.n_rollout):
        loss_list = []
        
        print("Rollout %d / %d" % (idx_episode, args.n_rollout))

        B = 1
        n_particle, n_shape = 0, 0

        # sampled particles
        datas = []
        gt_datas = []
        p_sample = [] 
        p_gt = []
        # s_gt = []
        for step in range(args.time_step):
            frame_name = str(step) + '.h5'
            gt_frame_name = 'gt_' + str(step) + '.h5'
            # gt_frame_name = str(step) + '.h5'
            if args.shape_aug:
                frame_name = 'shape_' + frame_name
                gt_frame_name = 'shape_' + gt_frame_name
                gt_frame_name = f'gttmp_{str(step)}.h5'
            
            data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), frame_name)
            gt_data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), gt_frame_name)
            
            data = load_data(data_names, data_path)
            gt_data = load_data(data_names, gt_data_path)
            
            if n_particle == 0 and n_shape == 0:
                n_particle, n_shape, scene_params = get_scene_info(data)
                # import pdb; pdb.set_trace()
                scene_params = torch.FloatTensor(scene_params).unsqueeze(0)

            if args.verbose_data:
                print("n_particle", n_particle)
                print("n_shape", n_shape)
            
            datas.append(data)
            gt_datas.append(gt_data)

            p_sample.append(data[0])
            p_gt.append(gt_data[0])
            # s_gt.append(data[1])
        # p_sample: time_step x N x state_dim
        # s_gt: time_step x n_s x 4
        p_sample = torch.FloatTensor(np.stack(p_sample))
        p_gt = torch.FloatTensor(np.stack(p_gt))
        # s_gt = torch.FloatTensor(np.stack(s_gt))
        p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)
        # initialize particle grouping
        group_info = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

        print('scene_params:', group_info[-1][0, 0].item())

        # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
        # for now, only used as a placeholder
        memory_init = model.init_memory(B, n_particle + n_shape)

        # model rollout
        loss = 0.
        loss_raw = 0.
        loss_counter = 0.
        st_idx = args.n_his
        ed_idx = args.time_step

        with torch.set_grad_enabled(False):
            for step_id in range(st_idx, ed_idx):
                # print(step_id)
                if step_id == st_idx:
                    if args.gt_particles:
                        # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                        state_cur = p_gt[step_id - args.n_his:step_id]
                    else:
                        state_cur = p_sample[step_id - args.n_his:step_id]
                    if use_gpu:
                        state_cur = state_cur.cuda()

                if step_id % 50 == 0:
                    print("Step %d / %d" % (step_id, ed_idx))

                # attr: (n_p + n_s) x attr_dim
                # Rr_cur, Rs_cur: n_rel x (n_p + n_s)
                # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                attr, _, Rr_cur, Rs_cur, cluster_onehot = prepare_input(state_cur[-1].cpu().numpy(), n_particle, n_shape, args, stdreg=args.stdreg)
                if use_gpu:
                    attr = attr.cuda()
                    Rr_cur = Rr_cur.cuda()
                    Rs_cur = Rs_cur.cuda()

                # t
                st_time = time.time()

                # unsqueeze the batch dimension
                # attr: B x (n_p + n_s) x attr_dim
                # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
                # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                attr = attr.unsqueeze(0)
                Rr_cur = Rr_cur.unsqueeze(0)
                Rs_cur = Rs_cur.unsqueeze(0)
                state_cur = state_cur.unsqueeze(0)
                if cluster_onehot:
                    cluster_onehot = cluster_onehot.unsqueeze(0)

                if args.stage in ['dy']:
                    inputs = [attr, state_cur, Rr_cur, Rs_cur, memory_init, group_info, cluster_onehot]
                # pred_pos (unnormalized): B x n_p x state_dim
                # pred_motion_norm (normalized): B x n_p x state_dim
                # import pdb; pdb.set_trace()
                if args.sequence_length > args.n_his+1:
                    pred_pos_p, pred_motion_norm, std_cluster = model.predict_dynamics(inputs, (step_id-args.n_his))
                else:
                    pred_pos_p, pred_motion_norm, std_cluster = model.predict_dynamics(inputs)
                # concatenate the state of the shapes
                # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                sample_pos = p_sample[step_id].unsqueeze(0)
                sample_pos_p = sample_pos[:, :n_particle]
                if use_gpu:
                    sample_pos = sample_pos.cuda()
                pred_pos = torch.cat([pred_pos_p, sample_pos[:, n_particle:]], 1)

                # sample_motion_norm (normalized): B x (n_p + n_s) x state_dim
                # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                sample_motion = (p_sample[step_id] - p_sample[step_id - 1]).unsqueeze(0)
                if use_gpu:
                    sample_motion = sample_motion.cuda()
                mean_d, std_d = model.stat[2:]
                sample_motion_norm = (sample_motion - mean_d) / std_d
                pred_motion_norm = torch.cat([pred_motion_norm, sample_motion_norm[:, n_particle:]], 1)

                loss_cur = F.l1_loss(pred_motion_norm[:, :n_particle], sample_motion_norm[:, :n_particle])
                loss_cur_raw = F.l1_loss(pred_pos_p, sample_pos_p)
                loss_emd = emd_loss(pred_pos_p, sample_pos_p)
                loss_chamfer = chamfer_loss(pred_pos_p, sample_pos_p)
                loss_uh = uh_loss(pred_pos_p, sample_pos_p)

                loss += loss_cur
                loss_raw += loss_cur_raw
                loss_counter += 1
                loss_list.append([step_id, loss_emd.item(), loss_chamfer.item(), loss_uh.item()])
                # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
                state_cur = state_cur.detach()[0]

                # record the prediction
                p_pred[step_id] = state_cur[-1].detach().cpu()

        '''
        print loss
        '''
        loss /= loss_counter
        loss_raw /= loss_counter
        print("loss: %.6f, loss_raw: %.10f" % (loss.item(), loss_raw.item()))

        loss_list_over_episodes.append(loss_list)
        # print(loss_list)
        # import pdb; pdb.set_trace()
        '''
        visualization
        '''
        group_info = [d.data.cpu().numpy()[0, ...] for d in group_info]
        if args.gt_particles:
            p_pred = np.concatenate((p_gt.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
        else:
            p_pred = np.concatenate((p_sample.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
        p_sample = p_sample.numpy()[:ed_idx]
        p_gt = p_gt.numpy()[:ed_idx]
        # s_gt = s_gt.numpy()[st_idx:ed_idx]
        vis_length = ed_idx - st_idx
        vid_path = os.path.join(args.dataf, 'vid', str(idx_episode).zfill(3))
        render_path = os.path.join(args.evalf, 'render', f'vid_{idx_episode}_plt.gif')

        if args.vis == 'plt':
            plt_render([p_gt, p_sample, p_pred], n_particle, vid_path, render_path)
        else:
            raise NotImplementedError

    # plot the loss curves for training and evaluating
    with open(os.path.join(args.outf, 'train.npy'), 'rb') as f:
        train_log = np.load(f, allow_pickle=True)
        train_log = train_log[None][0]
        train_plot_curves(train_log['iters'], train_log['loss'], path=os.path.join(args.evalf, 'plot', 'train_loss_curves.png'))

    loss_list_over_episodes = np.array(loss_list_over_episodes)

    eval_plot_curves(np.mean(loss_list_over_episodes, axis=0), path=os.path.join(args.evalf, 'plot', 'eval_loss_curves.png'))

    print(f"emd loss at last frame: {loss_list_over_episodes[:, -1, 1]}")

    info = f"\nAverage (+- std) emd loss over episodes: {np.mean(loss_list_over_episodes[:, :, 1])} (+- {np.std(loss_list_over_episodes[:, :, 1])})"
    info += f"\nAverage (+- std) chamfer loss over episodes: {np.mean(loss_list_over_episodes[:, :, 2])} (+- {np.std(loss_list_over_episodes[:, :, 2])})"
    info += f"\nAverage (+- std) hausdorff loss over episodes: {np.mean(loss_list_over_episodes[:, :, 3])} (+- {np.std(loss_list_over_episodes[:, :, 3])})"
    print(info)

if __name__ == '__main__':
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_eval) > 0:
        args.outf = args.outf_eval

    # with open(os.path.join(args.outf, 'train.npy'), 'rb') as f:
    #     train_log = np.load(f, allow_pickle=True)
    #     train_log = train_log[None][0]
    #     if 'args' in train_log:
    #         train_args = argparse.Namespace(**train_log['args'])
    #         args.gt_particles = train_args.gt_particles
            
    evaluate(args, args.eval_epoch, args.eval_iter)
