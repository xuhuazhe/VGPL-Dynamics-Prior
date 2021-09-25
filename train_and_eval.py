import os
import time
import sys
import copy

import multiprocessing as mp
from progressbar import ProgressBar

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import gen_args
from data_utils import PhysicsFleXDataset
from data_utils import prepare_input, get_scene_info, get_env_group
from models import Model, ChamferLoss, HausdorfLoss, EarthMoverLoss, UpdatedHausdorffLoss, ClipLoss
from utils import make_graph, check_gradient, set_seed, AverageMeter, get_lr, Tee
from utils import count_parameters, my_collate, matched_motion


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)
os.system('mkdir -p ' + args.evalf)
os.system('mkdir -p ' + os.path.join(args.evalf, 'render'))

tee_train = Tee(os.path.join(args.outf, 'train.log'), 'w')
tee_eval = Tee(os.path.join(args.evalf, 'eval.log'), 'w')

### training setup

# load training data

phases = ['train', 'valid'] if args.eval == 0 else ['valid']
datasets = {phase: PhysicsFleXDataset(args, phase) for phase in phases}

for phase in phases:
    if args.gen_data:
        datasets[phase].gen_data(args.env)
    else:
        datasets[phase].load_data(args.env)

dataloaders = {phase: DataLoader(
    datasets[phase],
    batch_size=args.batch_size,
    shuffle=True if phase == 'train' else False,
    num_workers=args.num_workers,
    collate_fn=my_collate) for phase in phases}

# create model and train
use_gpu = torch.cuda.is_available()
model = Model(args, use_gpu)

print("model #params: %d" % count_parameters(model))


# checkpoint to reload model from
model_path = None

# resume training of a saved model (if given)
if args.resume == 0:
    print("Randomly initialize the model's parameters")

elif args.resume == 1:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (
        args.resume_epoch, args.resume_iter))
    print("Loading saved ckp from %s" % model_path)

    if args.stage == 'dy':
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()

        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)


# optimizer
if args.stage == 'dy':
    params = model.dynamics_predictor.parameters()
else:
    raise AssertionError("unknown stage: %s" % args.stage)

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(
        params, lr=args.lr, betas=(args.beta1, 0.999))
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9)
else:
    raise AssertionError("unknown optimizer: %s" % args.optimizer)

# reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

# define loss
particle_dist_loss = ChamferLoss()   #torch.nn.L1Loss()
h_loss = HausdorfLoss()
emd_loss = EarthMoverLoss()
uh_loss = UpdatedHausdorffLoss()
clip_loss = ClipLoss()
if use_gpu:
    model = model.cuda()

# log args
print(args)

### eval setup
data_names = args.data_names

if args.eval_epoch < 0:
    model_name = 'net_best.pth'
else:
    model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter)

# model_dir = 'files_dy24-Sep-2021-11:06:41.989613_nHis4_aug0.05emd_seqlen5_uhw0.0_clipw0.0'
# model_path = os.path.join('dump/dump_Gripper/' + model_dir, model_name)    # args.outf
model_path = os.path.join(args.outf, model_name)
print("Loading network from %s" % model_path)


# start training
st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf

training_stats = {'loss':[], 'loss_raw':[], 'iters': [], 'loss_emd': [], 'loss_motion': []}
for epoch in range(st_epoch, args.n_epoch):

    for phase in phases:

        model.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_raw = AverageMeter()

        meter_loss_ref = AverageMeter()
        meter_loss_nxt = AverageMeter()

        meter_loss_param = AverageMeter()


        bar = ProgressBar(max_value=len(dataloaders[phase]))

        for i, data in bar(enumerate(dataloaders[phase])):
            # each "data" is a trajectory of sequence_length time steps

            if args.stage == 'dy':
                # attrs: B x (n_p + n_s) x attr_dim
                # particles: B x seq_length x (n_p + n_s) x state_dim
                # n_particles: B
                # n_shapes: B
                # scene_params: B x param_dim
                # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                attrs, particles, n_particles, n_shapes, scene_params, Rrs, Rss, cluster_onehots = data
                if use_gpu:
                    attrs = attrs.cuda()
                    particles = particles.cuda()
                    Rrs, Rss = Rrs.cuda(), Rss.cuda()
                    if cluster_onehots is not None:
                        cluster_onehots = cluster_onehots.cuda()

                # statistics
                B = attrs.size(0)
                n_particle = n_particles[0].item()
                n_shape = n_shapes[0].item()

                # p_rigid: B x n_instance
                # p_instance: B x n_particle x n_instance
                # physics_param: B x n_particle
                groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

                # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                # for now, only used as a placeholder
                memory_init = model.init_memory(B, n_particle + n_shape)
                loss = 0
                for j in range(args.sequence_length - args.n_his):
                    with torch.set_grad_enabled(phase == 'train'):
                        # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                        if j == 0:
                            state_cur = particles[:, :args.n_his]
                            # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                            Rr_cur = Rrs[:, args.n_his - 1]
                            Rs_cur = Rss[:, args.n_his - 1]
                        else: # elif pred_pos.size(0) >= args.batch_size:
                            Rr_cur = []
                            Rs_cur = []
                            max_n_rel = 0
                            for k in range(pred_pos.size(0)):
                                _, _, Rr_cur_k, Rs_cur_k, _ = prepare_input(pred_pos[k].detach().cpu().numpy(), n_particle, n_shape, args, stdreg=args.stdreg)
                                Rr_cur.append(Rr_cur_k)
                                Rs_cur.append(Rs_cur_k)
                                max_n_rel = max(max_n_rel, Rr_cur_k.size(0))
                            for w in range(pred_pos.size(0)):
                                Rr_cur_k, Rs_cur_k = Rr_cur[w], Rs_cur[w]
                                Rr_cur_k = torch.cat([Rr_cur_k, torch.zeros(max_n_rel - Rr_cur_k.size(0), n_particle + n_shape)], 0)
                                Rs_cur_k = torch.cat([Rs_cur_k, torch.zeros(max_n_rel - Rs_cur_k.size(0), n_particle + n_shape)], 0)
                                Rr_cur[w], Rs_cur[w] = Rr_cur_k, Rs_cur_k
                            Rr_cur = torch.FloatTensor(np.stack(Rr_cur))
                            Rs_cur = torch.FloatTensor(np.stack(Rs_cur))
                            if use_gpu:
                                Rr_cur = Rr_cur.cuda()
                                Rs_cur = Rs_cur.cuda()
                            state_cur = torch.cat([state_cur[:,-3:], pred_pos.unsqueeze(1)], dim=1)


                        if cluster_onehots is not None:
                            cluster_onehot = cluster_onehots[:, args.n_his - 1]
                        else:
                            cluster_onehot = None
                        # predict the velocity at the next time step
                        inputs = [attrs, state_cur, Rr_cur, Rs_cur, memory_init, groups_gt, cluster_onehot]

                        # pred_pos (unnormalized): B x n_p x state_dim
                        # pred_motion_norm (normalized): B x n_p x state_dim
                        pred_pos, pred_motion_norm, std_cluster = model.predict_dynamics(inputs)

                        # concatenate the state of the shapes
                        # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                        gt_pos = particles[:, args.n_his]
                        pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

                        # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                        # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                        # gt_motion_norm should match then calculate if matched_motion enabled
                        if args.matched_motion:
                            gt_motion = matched_motion(particles[:, args.n_his], particles[:, args.n_his - 1], n_particles=n_particle)
                        else:
                            gt_motion = particles[:, args.n_his] - particles[:, args.n_his - 1]

                        mean_d, std_d = model.stat[2:]
                        gt_motion_norm = (gt_motion - mean_d) / std_d
                        pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)
                        if args.losstype == 'emd':
                            loss += emd_loss(pred_pos, gt_pos)  #particle_dist_loss(pred_pos, gt_pos) + h_loss(pred_pos, gt_pos) #F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                        elif args.losstype == 'chamfer':
                            loss += particle_dist_loss(pred_pos, gt_pos)
                        elif args.losstype == 'hausdorff':
                            loss += particle_dist_loss(pred_pos, gt_pos) + h_loss(pred_pos, gt_pos)
                        elif args.losstype == 'l1':
                            loss += F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                        elif args.losstype == 'emd_uh':
                            loss_emd = emd_loss(pred_pos, gt_pos)
                            loss_uh = uh_loss(pred_pos, gt_pos)
                            # print(loss_emd, loss_uh)
                            loss += loss_emd + args.uh_weight * loss_uh
                        elif args.losstype == 'emd_l1':
                            loss_emd = emd_loss(pred_pos, gt_pos)
                            loss_motion = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                            # print('emd:', loss_emd.item())
                            # print('l1:', args.matched_motion_weight * loss_motion.item())
                            loss += loss_emd + args.matched_motion_weight * loss_motion
                        elif args.losstype == 'emd_uh_clip':
                            loss_emd = emd_loss(pred_pos, gt_pos)
                            loss_uh = uh_loss(pred_pos, gt_pos)
                            loss_clip = clip_loss(pred_pos, pred_pos) # self dist
                            print(loss_emd.item(), loss_uh.item(), loss_clip.item())
                            loss += loss_emd + args.uh_weight * loss_uh + args.clip_weight * loss_clip
                        else:
                            raise NotImplementedError

                        if args.stdreg:
                            loss += args.stdreg_weight * std_cluster
                        loss_raw = F.l1_loss(pred_pos, gt_pos)

                        meter_loss.update(loss.item(), B)
                        meter_loss_raw.update(loss_raw.item(), B)

            if i % args.log_per_iter == 0:
                print()
                print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                    phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                    loss.item(), meter_loss.avg, loss_raw.item(), meter_loss_raw.avg))
                print('std_cluster', std_cluster)
                if phase == 'train':
                    training_stats['loss'].append(loss.item())
                    training_stats['loss_raw'].append(loss_raw.item())
                    training_stats['iters'].append(epoch * len(dataloaders[phase]) + i)
                    if args.losstype == 'emd_l1':
                        training_stats['loss_emd'].append(loss_emd.item())
                        training_stats['loss_motion'].append(loss_motion.item())
                # with open(args.outf + '/train.npy', 'wb') as f:
                #     np.save(f, training_stats)

            # update model parameters
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if phase == 'train' and i > 0 and ((epoch * len(dataloaders[phase])) + i) % args.ckp_per_iter == 0:
                model_path = '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                torch.save(model.state_dict(), model_path)



        print('%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss))

        with open(args.outf + '/train.npy','wb') as f:
            np.save(f, training_stats)

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))

### evaluating

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


infos = np.arange(50)
emd_loss = EarthMoverLoss()
uh_loss = UpdatedHausdorffLoss()
for idx_episode in range(0, 50, 1): #range(len(infos)):
    emd_list = []
    print("Rollout %d / %d" % (idx_episode, len(infos)))

    B = 1
    n_particle, n_shape = 0, 0

    # ground truth
    datas = []
    p_gt = []
    s_gt = []
    for step in range(args.time_step):
        data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), str(step) + '.h5')

        data = load_data(data_names, data_path)
        if n_particle == 0 and n_shape == 0:
            n_particle, n_shape, scene_params = get_scene_info(data)
            # import pdb; pdb.set_trace()
            scene_params = torch.FloatTensor(scene_params).unsqueeze(0)

        if args.verbose_data:
            print("n_particle", n_particle)
            print("n_shape", n_shape)
        datas.append(data)

        p_gt.append(data[0])
        s_gt.append(data[1])
    # p_gt: time_step x N x state_dim
    # s_gt: time_step x n_s x 4
    p_gt = torch.FloatTensor(np.stack(p_gt))
    s_gt = torch.FloatTensor(np.stack(s_gt))
    p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)
    # initialize particle grouping
    group_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

    print('scene_params:', group_gt[-1][0, 0].item())

    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
    # for now, only used as a placeholder
    memory_init = model.init_memory(B, n_particle + n_shape)

    # model rollout
    loss = 0.
    loss_raw = 0.
    loss_counter = 0.
    st_idx = args.n_his
    ed_idx = args.sequence_length

    with torch.set_grad_enabled(False):

        for step_id in range(st_idx, ed_idx):

            if step_id == st_idx:
                # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                state_cur = p_gt[step_id - args.n_his:step_id]
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
                inputs = [attr, state_cur, Rr_cur, Rs_cur, memory_init, group_gt, cluster_onehot]
            # pred_pos (unnormalized): B x n_p x state_dim
            # pred_motion_norm (normalized): B x n_p x state_dim
            pred_pos, pred_motion_norm, std_cluster = model.predict_dynamics(inputs)

            # concatenate the state of the shapes
            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
            gt_pos = p_gt[step_id].unsqueeze(0)
            if use_gpu:
                gt_pos = gt_pos.cuda()
            pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
            gt_motion = (p_gt[step_id] - p_gt[step_id - 1]).unsqueeze(0)
            if use_gpu:
                gt_motion = gt_motion.cuda()
            mean_d, std_d = model.stat[2:]
            gt_motion_norm = (gt_motion - mean_d) / std_d
            pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)

            loss_cur = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
            loss_cur_raw = F.l1_loss(pred_pos, gt_pos)
            loss_emd = emd_loss(pred_pos, gt_pos)
            loss_uh = uh_loss(pred_pos, gt_pos)

            loss += loss_cur
            loss_raw += loss_cur_raw
            loss_counter += 1
            emd_list.append((step_id, loss_emd.item(), loss_uh.item()))
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

    plot_curves(emd_list)
    # import pdb; pdb.set_trace()
    '''
    visualization
    '''
    group_gt = [d.data.cpu().numpy()[0, ...] for d in group_gt]
    p_pred = p_pred.numpy()[st_idx:ed_idx]
    p_gt = p_gt.numpy()[st_idx:ed_idx]
    s_gt = s_gt.numpy()[st_idx:ed_idx]
    vis_length = ed_idx - st_idx
    # print(vis_length)

    if args.vispy:

        ### render in VisPy
        import vispy.scene
        from vispy import app
        from vispy.visuals import transforms

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


        c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
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
        #1 - p_gt: seq_length x n_p x 3
        #2 - s_gt: seq_length x n_s x 3
        print('p_pred', p_pred.shape)
        print('p_gt', p_gt.shape)
        print('s_gt', s_gt.shape)
        # create directory to save images if not exist
        vispy_dir = args.evalf + "/vispy"
        os.system('mkdir -p ' + vispy_dir)


        def update(event):
            global p1
            global t_step
            global colors

            if t_step < vis_length:
                if t_step == 0:
                    print("Rendering ground truth")

                t_actual = t_step

                colors = convert_groups_to_colors(
                    group_gt, n_particle, args.n_instance,
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


            elif vis_length <= t_step < vis_length * 2:
                if t_step == vis_length:
                    print("Rendering prediction result")

                t_actual = t_step - vis_length

                colors = convert_groups_to_colors(
                    group_gt, n_particle, args.n_instance,
                    instance_colors=instance_colors, env=args.env)

                colors = np.clip(colors, 0., 1.)

                if args.env == "Gripper":
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

            else:
                # discarded frames
                pass

            # time forward
            t_step += 1


        # start animation
        timer = app.Timer()
        timer.connect(update)
        timer.start(interval=1. / 60., iterations=vis_length * 2)

        c.show()
        app.run()