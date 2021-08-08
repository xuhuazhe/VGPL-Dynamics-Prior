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
from models import Model, ChamferLoss, HausdorfLoss, EarthMoverLoss
from utils import make_graph, check_gradient, set_seed, AverageMeter, get_lr, Tee
from utils import count_parameters, my_collate, matched_motion


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)

tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


### training

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
if use_gpu:
    model = model.cuda()

# log args
print(args)

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

                with torch.set_grad_enabled(phase == 'train'):
                    # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                    state_cur = particles[:, :args.n_his]

                    # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                    Rr_cur = Rrs[:, args.n_his - 1]
                    Rs_cur = Rss[:, args.n_his - 1]
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
                        loss = emd_loss(pred_pos, gt_pos)  #particle_dist_loss(pred_pos, gt_pos) + h_loss(pred_pos, gt_pos) #F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                    elif args.losstype == 'chamfer':
                        loss = particle_dist_loss(pred_pos, gt_pos)
                    elif args.losstype == 'hausdorff':
                        loss = particle_dist_loss(pred_pos, gt_pos) + h_loss(pred_pos, gt_pos)
                    elif args.losstype == 'l1':
                        loss = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                    elif args.losstype == 'emd_l1':
                        loss_emd = emd_loss(pred_pos, gt_pos)
                        loss_motion = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
                        # print('emd:', loss_emd.item())
                        # print('l1:', args.matched_motion_weight * loss_motion.item())
                        loss = loss_emd + args.matched_motion_weight * loss_motion
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
