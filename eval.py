import argparse
import copy
import os
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

import vispy
from sys import platform
if platform != 'darwin':
    vispy.use('osmesa')
import vispy.scene
from vispy import app
from vispy.visuals import transforms

import cProfile
import pstats
import io

# render particles
p1 = vispy.scene.visuals.Markers()
p2 = vispy.scene.visuals.Markers()
p1.antialias = 0  # remove white edge

# set animation
t_step = 0

def evaluate(args, eval_epoch, eval_iter):
    global p1
    global t_step
    global colors
    
    args.evalf = os.path.join(args.outf, 'eval')

    os.system('mkdir -p ' + args.evalf)
    os.system('mkdir -p ' + os.path.join(args.evalf, 'plot'))
    os.system('mkdir -p ' + os.path.join(args.evalf, 'render'))
    os.system('mkdir -p ' + os.path.join(args.evalf, 'vispy'))

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
            shape_frame_name = 'shape_gt_' + frame_name
            if args.shape_aug:
                data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), shape_frame_name)
            else:
                data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), frame_name)

            gt_data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), 'gt_' + str(step) + '.h5')
            
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
                    pred_pos, pred_motion_norm, std_cluster = model.predict_dynamics(inputs, (step_id-args.n_his))
                else:
                    pred_pos, pred_motion_norm, std_cluster = model.predict_dynamics(inputs)
                # concatenate the state of the shapes
                # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                sample_pos = p_sample[step_id].unsqueeze(0)
                if use_gpu:
                    sample_pos = sample_pos.cuda()
                pred_pos = torch.cat([pred_pos, sample_pos[:, n_particle:]], 1)

                # sample_motion_norm (normalized): B x (n_p + n_s) x state_dim
                # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                sample_motion = (p_sample[step_id] - p_sample[step_id - 1]).unsqueeze(0)
                if use_gpu:
                    sample_motion = sample_motion.cuda()
                mean_d, std_d = model.stat[2:]
                sample_motion_norm = (sample_motion - mean_d) / std_d
                pred_motion_norm = torch.cat([pred_motion_norm, sample_motion_norm[:, n_particle:]], 1)

                loss_cur = F.l1_loss(pred_motion_norm[:, :n_particle], sample_motion_norm[:, :n_particle])
                loss_cur_raw = F.l1_loss(pred_pos, sample_pos)
                loss_emd = emd_loss(pred_pos, sample_pos)
                loss_chamfer = chamfer_loss(pred_pos, sample_pos)
                loss_uh = uh_loss(pred_pos, sample_pos)

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
        # print(vis_length)

        if args.vispy:

            ### render in VisPy
            particle_size = 0.01
            border = 0.025
            height = 1.3
            y_rotate_deg = -45.0

            def y_rotate(obj, deg=y_rotate_deg):
                tr = vispy.visuals.transforms.MatrixTransform()
                tr.rotate(deg, (0, 1, 0))
                obj.transform = tr

            def add_floor(v):
                # add floor
                floor_length = 3.0
                w, h, d = floor_length, floor_length, border
                b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
                y_rotate(b1)
                v.add(b1)

                # adjust position of box
                mesh_b1 = b1.mesh.mesh_data
                v1 = mesh_b1.get_vertices()
                c1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
                mesh_b1.set_vertices(np.add(v1, c1))

                mesh_border_b1 = b1.border.mesh_data
                vv1 = mesh_border_b1.get_vertices()
                cc1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
                mesh_border_b1.set_vertices(np.add(vv1, cc1))

            def update_box_states(boxes, last_states, curr_states):
                v = curr_states[0] - last_states[0]
                if args.verbose_data:
                    print("box states:", last_states, curr_states)
                    print("box velocity:", v)

                tr = vispy.visuals.transforms.MatrixTransform()
                tr.rotate(y_rotate_deg, (0, 1, 0))

                for i, box in enumerate(boxes):
                    # use v to update box translation
                    trans = (curr_states[i][0], curr_states[i][1], curr_states[i][2])
                    box.transform = tr * vispy.visuals.transforms.STTransform(translate=trans)

            def translate_box(b, x, y, z):
                mesh_b = b.mesh.mesh_data
                v = mesh_b.get_vertices()
                c = np.array([x, y, z], dtype=np.float32)
                mesh_b.set_vertices(np.add(v, c))

                mesh_border_b = b.border.mesh_data
                vv = mesh_border_b.get_vertices()
                cc = np.array([x, y, z], dtype=np.float32)
                mesh_border_b.set_vertices(np.add(vv, cc))

            def add_box(v, w=0.1, h=0.1, d=0.1, x=0.0, y=0.0, z=0.0):
                """
                Add a box object to the scene view
                :param v: view to which the box should be added
                :param w: width
                :param h: height
                :param d: depth
                :param x: x center
                :param y: y center
                :param z: z center
                :return: None
                """
                # render background box
                b = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
                y_rotate(b)
                v.add(b)

                # adjust position of box
                translate_box(b, x, y, z)

                return b

            def calc_box_init(x, z):
                boxes = []

                # floor
                boxes.append([x, z, border, 0., -particle_size / 2, 0.])

                # left wall
                boxes.append([border, z, (height + border), -particle_size / 2, 0., 0.])

                # right wall
                boxes.append([border, z, (height + border), particle_size / 2, 0., 0.])

                # back wall
                boxes.append([(x + border * 2), border, (height + border)])

                # front wall (disabled when colored)
                # boxes.append([(x + border * 2), border, (height + border)])

                return boxes

            def add_container(v, box_x, box_z):
                boxes = calc_box_init(box_x, box_z)
                visuals = []
                for b in boxes:
                    if len(b) == 3:
                        visual = add_box(v, b[0], b[1], b[2])
                    elif len(b) == 6:
                        visual = add_box(v, b[0], b[1], b[2], b[3], b[4], b[5])
                    else:
                        raise AssertionError("Input should be either length 3 or length 6")
                    visuals.append(visual)
                return visuals


            c = vispy.scene.SceneCanvas(show=True, size=(512, 512), bgcolor='white')
            view = c.central_widget.add_view()


            if args.env == 'Pinch':
                view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=90, elevation=20, distance=2, up='+y')
                # set instance colors
                instance_colors = create_instance_colors(args.n_instance)

                # render floor
                add_floor(view)

            if args.env == 'Gripper':
                view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=90, elevation=50, distance=2, up='+y')
                # set instance colors
                instance_colors = create_instance_colors(args.n_instance)

                # render floor
                add_floor(view)

            if args.env == 'RigidFall':
                view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=45, elevation=20, distance=2, up='+y')
                # set instance colors
                instance_colors = create_instance_colors(args.n_instance)

                # render floor
                add_floor(view)

            if args.env == 'MassRope':
                view.camera = vispy.scene.cameras.TurntableCamera(fov=30, azimuth=0, elevation=20, distance=8, up='+y')

                # set instance colors
                n_string_particles = 15
                instance_colors = create_instance_colors(args.n_instance)

                # render floor
                add_floor(view)

            # render particles
            p1 = vispy.scene.visuals.Markers()
            p2 = vispy.scene.visuals.Markers()
            p1.antialias = 0  # remove white edge

            y_rotate(p1)

            view.add(p1)

            # set animation
            t_step = 0

            '''
            set up data for rendering
            '''
            #0 - p_pred: seq_length x n_p x 3
            #1 - p_sample: seq_length x n_p x 3
            #2 - s_gt: seq_length x n_s x 3
            print('p_gt', p_gt.shape)
            print('p_sample', p_sample.shape)
            print('p_pred', p_pred.shape)
            # print('s_gt', s_gt.shape)
            # create directory to save images if not exist
            vispy_dir = args.evalf + "/vispy"
            os.system('mkdir -p ' + vispy_dir)

            def update(event):
                global p1
                global t_step
                global colors

                if t_step < vis_length:
                    if t_step == 0:
                        print("Rendering ground truth particles")

                    t_actual = t_step

                    colors = convert_groups_to_colors(
                        group_info, n_particle, args.n_instance,
                        instance_colors=instance_colors, env=args.env)
                    colors = np.clip(colors, 0., 1.)
                    # import pdb; pdb.set_trace()
                    if args.env == "Gripper":
                        new_p = np.delete(copy.copy(p_gt), -3, axis=1)
                        colors = np.concatenate([colors, np.array([[0, 1, 0, 1], [0, 1, 0, 1]])])
                    else:
                        new_p = np.delete(copy.copy(p_gt), -2, axis=1)
                        colors = np.concatenate([colors, np.array([[0,1,0,1]])])
                    # print('color shape!!!', colors.shape)
                    p1.set_data(new_p[t_actual], edge_color='black', face_color=colors)
                    # p1.set_data(p_gt[t_actual, :n_particle], edge_color='black', face_color=colors)
                    # p1.set_data(p_gt[t_actual, -1], edge_color='k', face_color='b')
                    # render for ground truth
                    img = c.render()
                    img_path = os.path.join(vispy_dir, "gt_{}_{}.png".format(str(idx_episode), str(t_actual)))
                    vispy.io.write_png(img_path, img)

                elif t_step < 2 * vis_length:
                    # import pdb; pdb.set_trace()
                    if t_step == vis_length:
                        print("Rendering sampled particles")

                    t_actual = t_step - vis_length

                    colors = convert_groups_to_colors(
                        group_info, n_particle, args.n_instance,
                        instance_colors=instance_colors, env=args.env)
                    colors = np.clip(colors, 0., 1.)
                    # import pdb; pdb.set_trace()
                    if args.env == "Gripper":
                        if args.shape_aug:
                            new_p = copy.copy(p_sample)
                            colors = np.concatenate([colors, np.array([[0, 1, 0, 1]]).repeat(31, axis=0) ] )
                        else:
                            new_p = np.delete(copy.copy(p_sample), -3, axis=1)
                            colors = np.concatenate([colors, np.array([[0, 1, 0, 1], [0, 1, 0, 1]])])
                    else:
                        new_p = np.delete(copy.copy(p_sample), -2, axis=1)
                        colors = np.concatenate([colors, np.array([[0,1,0,1]])])
                    # print('color shape!!!', colors.shape)
                    p1.set_data(new_p[t_actual], edge_color='black', face_color=colors)
                    # p1.set_data(p_sample[t_actual, :n_particle], edge_color='black', face_color=colors)
                    # p1.set_data(p_sample[t_actual, -1], edge_color='k', face_color='b')
                    # render for ground truth
                    img = c.render()
                    img_path = os.path.join(vispy_dir, "sample_{}_{}.png".format(str(idx_episode), str(t_actual)))
                    vispy.io.write_png(img_path, img)

                elif t_step < 3 * vis_length:
                    if t_step == 2 * vis_length:
                        print("Rendering prediction result")

                    t_actual = t_step - 2 * vis_length

                    colors = convert_groups_to_colors(
                        group_info, n_particle, args.n_instance,
                        instance_colors=instance_colors, env=args.env)

                    colors = np.clip(colors, 0., 1.)

                    if args.env == "Gripper":
                        if args.shape_aug:
                            new_p = copy.copy(p_pred)
                            colors = np.concatenate([colors, np.array([[0, 1, 0, 1]]).repeat(31, axis=0) ] )
                        else:
                            new_p = np.delete(copy.copy(p_pred), -3, axis=1)
                            colors = np.concatenate([colors, np.array([[0, 1, 0, 1], [0, 1, 0, 1]])])
                    else:
                        new_p = np.delete(copy.copy(p_pred), -2, axis=1)
                        colors = np.concatenate([colors, np.array([[0,1,0,1]])])
                    # new_p = np.delete(copy.copy(p_pred), -2, axis=1)
                    # colors = np.concatenate([colors, np.array([[0, 1, 0, 1]])])
                    p1.set_data(new_p[t_actual], edge_color='black', face_color=colors)
                    # p1.set_data(p_pred[t_actual, :n_particle], edge_color='black', face_color=colors)

                    # render for perception result
                    img = c.render()
                    img_path = os.path.join(vispy_dir, "pred_{}_{}.png".format(str(idx_episode), str(t_actual)))
                    vispy.io.write_png(img_path, img)

                    if t_step == vis_length * 3 - 1:
                        c.close()
                else:
                    # discarded frames
                    pass

                # time forward
                t_step += 1

            # start animation
            timer = app.Timer()
            timer.connect(update)
            timer.start(interval=1. / 60., iterations=vis_length * 3)

            c.show()
            app.run()

            # render video for evaluating grouping result
            if args.stage in ['dy']:
                print("Render video for dynamics prediction")

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(
                    os.path.join(args.evalf, 'render', 'vid_%d_vispy.avi' % (idx_episode)),
                    fourcc, 20, (1024, 1024))

                for step in range(vis_length):
                    vid_path = os.path.join(args.dataf, 'vid', str(idx_episode).zfill(3), str(step).zfill(3) + '_rgb_0.png')
                    gt_path = os.path.join(args.evalf, 'vispy', 'gt_%d_%d.png' % (idx_episode, step))
                    sample_path = os.path.join(args.evalf, 'vispy', 'sample_%d_%d.png' % (idx_episode, step))
                    pred_path = os.path.join(args.evalf, 'vispy', 'pred_%d_%d.png' % (idx_episode, step))

                    vid = cv2.imread(vid_path)
                    gt = cv2.imread(gt_path)
                    sample = cv2.imread(sample_path)
                    pred = cv2.imread(pred_path)

                    frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
                    try:
                        frame[:512, :512] = vid
                    except:
                        pass
                    frame[:512, 512:] = gt
                    frame[512:, :512] = sample
                    frame[512:, 512:] = pred

                    out.write(frame)

                out.release()

    # plot the loss curves for training and evaluating
    with open(os.path.join(args.outf, 'train.npy'), 'rb') as f:
        train_log = np.load(f, allow_pickle=True)
        train_log = train_log[None][0]
        train_plot_curves(train_log['iters'], train_log['loss'], path=os.path.join(args.evalf, 'plot', 'train_loss_curves.png'))

    loss_list_over_episodes = np.array(loss_list_over_episodes)

    eval_plot_curves(np.mean(loss_list_over_episodes, axis=0), path=os.path.join(args.evalf, 'plot', 'eval_loss_curves.png'))

    info = f"\nAverage (+- std) emd loss over episodes: {np.mean(loss_list_over_episodes[:, :, 1])} (+- {np.std(loss_list_over_episodes[:, :, 1])})"
    info += f"\nAverage (+- std) chamfer loss over episodes: {np.mean(loss_list_over_episodes[:, :, 2])} (+- {np.std(loss_list_over_episodes[:, :, 2])})"
    info += f"\nAverage (+- std) hausdorff loss over episodes: {np.mean(loss_list_over_episodes[:, :, 3])} (+- {np.std(loss_list_over_episodes[:, :, 3])})"
    print(info)

if __name__ == '__main__':
    args = gen_args()
    set_seed(args.random_seed)

    if len(args.outf_eval) > 0:
        args.outf = args.outf_eval

    with open(os.path.join(args.outf, 'train.npy'), 'rb') as f:
        train_log = np.load(f, allow_pickle=True)
        train_log = train_log[None][0]
        if 'args' in train_log:
            train_args = argparse.Namespace(**train_log['args'])
            args.gt_particles = train_args.gt_particles
            
    evaluate(args, args.eval_epoch, args.eval_iter)
