import argparse
import numpy as np
import torch
from datetime import datetime


### build arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', default='RigidFall')
parser.add_argument('--stage', default='dy', help="dy: dynamics model")
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--random_seed', type=int, default=42)

parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1. / 60.)
parser.add_argument('--n_instance', type=int, default=-1)

parser.add_argument('--nf_relation', type=int, default=150)
parser.add_argument('--nf_particle', type=int, default=150)
parser.add_argument('--nf_pos', type=int, default=150)
parser.add_argument('--nf_memory', type=int, default=150)
parser.add_argument('--mem_nlayer', type=int, default=2)
parser.add_argument('--nf_effect', type=int, default=150)
parser.add_argument('--losstype', type=str, default='l1')
parser.add_argument('--stdreg', type=int, default=0)
parser.add_argument('--stdreg_weight', type=float, default=0.0)
parser.add_argument('--matched_motion', type=int, default=0)
parser.add_argument('--matched_motion_weight', type=float, default=0.0)
parser.add_argument('--uh_weight', type=float, default=0.0)
parser.add_argument('--clip_weight', type=float, default=0.0)


parser.add_argument('--outf', default='files')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--dataf', default='data')
parser.add_argument('--data_type', type=str, default='none')

parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-6)

# for ablation study
parser.add_argument('--neighbor_radius', type=float, default=-1)
parser.add_argument('--neighbor_k', type=float, default=-1)

# use a flexible number of frames for each training iteration
parser.add_argument('--n_his', type=int, default=4)
parser.add_argument('--sequence_length', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

# physics parameter
parser.add_argument('--physics_param_range', type=float, nargs=2, default=None)

# width and height for storing vision
parser.add_argument('--vis_width', type=int, default=160)
parser.add_argument('--vis_height', type=int, default=120)


'''
train
'''

parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--log_per_iter', type=int, default=50)
parser.add_argument('--ckp_per_iter', type=int, default=1000)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--optimizer', default='Adam', help='Adam|SGD')
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=1)

# data generation
parser.add_argument('--gen_data', type=int, default=0)
parser.add_argument('--gen_stat', type=int, default=0)
parser.add_argument('--gen_vision', type=int, default=0)

parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--resume_epoch', type=int, default=-1)
parser.add_argument('--resume_iter', type=int, default=-1)

# data augmentation
parser.add_argument('--augment_ratio', type=float, default=0.)


'''
eval
'''
parser.add_argument('--eval_epoch', type=int, default=-1, help='pretrained model')
parser.add_argument('--eval_iter', type=int, default=-1, help='pretrained model')
parser.add_argument('--eval_set', default='demo')

# visualization flog
parser.add_argument('--pyflex', type=int, default=1)
parser.add_argument('--vispy', type=int, default=1)


def gen_args():
    args = parser.parse_args()

    args.data_names = ['positions', 'shape_quats', 'scene_params']

    if args.env == 'Pinch':
        args.env_idx = 1000

        args.n_rollout = 50
        args.time_step = 49

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, floor, prim]
        args.attr_dim = 3

        args.neighbor_radius = 0.05
        args.neighbor_k = 20

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 1
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (-5., -5.)

        args.outf = 'dump/dump_Pinch/' + args.outf + '_' + args.stage + suffix + datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        args.evalf = 'dump/dump_Pinch/' + args.evalf + '_' + args.stage + suffix# + datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

        args.mean_p = np.array([0.49868582, 0.11530433, 0.49752659])
        args.std_p = np.array([0.06167904, 0.05326168, 0.06180995])
        # args.mean_d = np.array([-1.62756886e-05, -1.10265409e-04, -1.71767924e-04])
        # args.std_d = np.array([0.08455442, 0.07277832, 0.08571255])
        args.mean_d = np.array([4.00109732e-04, 6.71352200e-05, -1.17460513e-04])
        args.std_d = np.array([0.0166535,  0.01636565, 0.01657304])

    elif args.env == 'Gripper':
        args.env_idx = 1001

        args.n_rollout = 50
        if args.data_type == 'ngrip':
            args.time_step = 59
        else:
            args.time_step = 49

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, floor, prim]
        args.attr_dim = 3

        args.neighbor_radius = 0.05
        args.neighbor_k = 20

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 1
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (-5., -5.)

        args.outf = 'dump/dump_Pinch/' + args.outf + '_' + args.stage + suffix + datetime.now().strftime(
            "%d-%b-%Y-%H:%M:%S.%f")
        args.evalf = 'dump/dump_Pinch/' + args.evalf + '_' + args.stage + suffix  # + datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

        args.mean_p = np.array([0.50932539, 0.11348496, 0.49837578])
        args.std_p = np.array([0.06474939, 0.04888084, 0.05906044])
        # args.mean_d = np.array([-1.62756886e-05, -1.10265409e-04, -1.71767924e-04])
        # args.std_d = np.array([0.08455442, 0.07277832, 0.08571255])
        args.mean_d = np.array([-0.00284736, 0.00286124, -0.00130389])
        args.std_d = np.array([0.01755744, 0.01663332, 0.01677678])

    elif args.env == 'RigidFall':
        args.env_idx = 3

        args.n_rollout = 5000
        args.time_step = 121

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, floor]
        args.attr_dim = 2

        args.neighbor_radius = 0.08
        args.neighbor_k = 20

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 3
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (-15., -5.)

        args.outf = 'dump/dump_RigidFall/' + args.outf + '_' + args.stage + suffix
        args.evalf = 'dump/dump_RigidFall/' + args.evalf + '_' + args.stage + suffix

        args.mean_p = np.array([0.14778039, 0.15373468, 0.10396217])
        args.std_p = np.array([0.27770899, 0.13548609, 0.15006677])
        args.mean_d = np.array([-1.91248869e-05, -2.05043765e-03, 2.10580908e-05])
        args.std_d = np.array([0.00468072, 0.00703023, 0.00304786])

    elif args.env == 'MassRope':
        args.env_idx = 9

        args.n_rollout = 3000
        args.time_step = 201

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, pin]
        args.attr_dim = 2

        args.neighbor_radius = 0.25
        args.neighbor_k = -1

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 2
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (0.25, 1.2)

        args.outf = 'dump/dump_MassRope/' + args.outf + '_' + args.stage + suffix
        args.evalf = 'dump/dump_MassRope/' + args.evalf + '_' + args.stage + suffix

        args.mean_p = np.array([0.06443707, 1.09444374, 0.04942945])
        args.std_p = np.array([0.45214754, 0.29002383, 0.41175843])
        args.mean_d = np.array([-0.00097918, -0.00033966, -0.00080952])
        args.std_d = np.array([0.02086366, 0.0145161, 0.01856096])

    else:
        raise AssertionError("Unsupported env")


    # path to data
    if args.data_type != 'none':
        args.dataf = 'data/' + args.dataf + '_' + args.data_type    #+ '_' + args.env
    else:
        args.dataf = 'data/' + args.dataf + '_' + args.env

    # n_his
    args.outf += '_nHis%d' % args.n_his
    args.evalf += '_nHis%d' % args.n_his


    # data augmentation
    if args.augment_ratio > 0:
        args.outf += '_aug%.2f' % args.augment_ratio
        args.evalf += '_aug%.2f' % args.augment_ratio

    args.outf += args.losstype
    args.outf += f'_seqlen{args.sequence_length}'
    args.outf += f'_uhw{args.uh_weight}'
    args.outf += f'_clipw{args.clip_weight}'


    # evaluation checkpoints
    if args.stage in ['dy']:
        if args.eval_epoch > -1:
            args.evalf += '_dyEpoch_' + str(args.eval_epoch)
            args.evalf += '_dyIter_' + str(args.eval_iter)
        else:
            args.evalf += '_dyEpoch_best'

        args.evalf += '_%s' % args.eval_set


    return args
