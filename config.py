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

parser.add_argument('--stdreg', type=int, default=0)
parser.add_argument('--stdreg_weight', type=float, default=0.0)
parser.add_argument('--matched_motion', type=int, default=0)
parser.add_argument('--matched_motion_weight', type=float, default=0.0)

parser.add_argument('--valid', type=int, default=0)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-6)

# file paths
parser.add_argument('--outf', default='files')
parser.add_argument('--outf_eval', default='')
parser.add_argument('--outf_control', default='')
parser.add_argument('--outf_new', default='')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--dataf', default='data')
parser.add_argument('--gripperf', default='../PlasticineLab/plb/envs/gripper_fixed.yml')

# for ablation study
parser.add_argument('--neighbor_radius', type=float, default=-1)
parser.add_argument('--gripper_extra_neighbor_radius', type=float, default=-1)
parser.add_argument('--neighbor_k', type=float, default=-1)

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
parser.add_argument('--data_type', type=str, default='none')
parser.add_argument('--gt_particles', type=int, default=0)
parser.add_argument('--shape_aug', type=int, default=0)

parser.add_argument('--loss_type', type=str, default='l1')
parser.add_argument('--uh_weight', type=float, default=0.0)
parser.add_argument('--clip_weight', type=float, default=0.0)
parser.add_argument('--emd_weight', type=float, default=0.0)
parser.add_argument('--chamfer_weight', type=float, default=0.0)
parser.add_argument('--p_rigid', type=float, default=1.)
parser.add_argument('--alpha', type=float, default=0.05)

# use a flexible number of frames for each training iteration
parser.add_argument('--n_his', type=int, default=4)
parser.add_argument('--sequence_length', type=int, default=0)

parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--log_per_iter', type=int, default=50)
parser.add_argument('--ckp_per_iter', type=int, default=1000)

parser.add_argument('--n_epoch', type=int, default=100) # 100 FOR TEST, *1000* 
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
parser.add_argument('--eval_set', default='train')

# visualization flog
parser.add_argument('--pyflex', type=int, default=1)
parser.add_argument('--vis', type=str, default='plt')


'''
control
'''
parser.add_argument('--opt_algo', type=str, default='max')
parser.add_argument('--control_algo', type=str, default='fix')
parser.add_argument('--predict_horizon', type=int, default=2)
parser.add_argument('--control_sample_size', type=int, default=8)
parser.add_argument('--control_batch_size', type=int, default=4)
parser.add_argument('--reward_type', type=str, default='emd')
parser.add_argument('--use_sim', type=int, default=0)
parser.add_argument('--gt_action', type=int, default=0)
parser.add_argument('--gt_state_goal', type=int, default=0)
parser.add_argument('--subgoal', type=int, default=0)
parser.add_argument('--correction', type=int, default=0)
parser.add_argument('--n_grips', type=int, default=3)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--shape_type', type=str, default='')
parser.add_argument('--goal_shape_name', type=str, default='')
parser.add_argument('--CEM_opt_iter', type=int, default=1)
parser.add_argument('--CEM_init_pose_sample_size', type=int, default=40)
parser.add_argument('--CEM_gripper_rate_sample_size', type=int, default=8)
parser.add_argument('--GD_batch_size', type=int, default=1)
parser.add_argument('--sample_method', type=str, default="random")

### only useful for rl
parser.add_argument("--algo", type=str, default='sac')
parser.add_argument("--env_name", type=str, default="gripper_fixed-v1")
parser.add_argument("--path", type=str, default='./tmp')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_steps", type=int, default=None)

# differentiable physics parameters
parser.add_argument("--rllr", type=float, default=0.1)
parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
parser.add_argument("--outf_rl", type=str, default='')
# parser.add_argument("--gripperf", type=str, default="../PlasticineLab/plb/envs/gripper_fixed.yml")



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
        args.mean_d = np.array([0.0, 0.0, 0.0])
        args.std_d = np.array([1.0, 1.0, 1.0])

    elif args.env == 'Gripper':
        args.env_idx = 1001

        args.time_step = 119

        # object states:
        # [x, y, z]
        args.state_dim = 3

        # object attr:
        # [particle, floor, prim]
        args.attr_dim = 3

        args.neighbor_radius = 0.05
        
        if 'small' in args.data_type:
            args.gripper_extra_neighbor_radius = 0.0
        elif 'robot' in args.data_type:
            args.gripper_extra_neighbor_radius = 0.015
        else:
            args.gripper_extra_neighbor_radius = 0.015

        args.neighbor_k = 20

        suffix = ''
        if args.n_instance == -1:
            args.n_instance = 1
        else:
            suffix += '_nIns_' + str(args.n_instance)

        args.physics_param_range = (-5., -5.)

        args.outf =  f'dump/dump_{args.data_type}/{args.outf}_{args.stage}{suffix}_{datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")}'
        args.evalf = f'dump/dump_{args.data_type}/{args.evalf}_{args.stage}{suffix}'

        # if 'robot' in args.data_type:
        #     args.mean_p = np.array([0.50932539, 0.11348496, 0.49837578])
        #     args.std_p = np.array([0.06474939, 0.04888084, 0.05906044])
        # else:
        args.mean_p = np.array([0.50932539, 0.11348496, 0.49837578])
        args.std_p = np.array([0.06474939, 0.04888084, 0.05906044])
        
        # args.mean_d = np.array([-0.00284736, 0.00286124, -0.00130389])
        # args.std_d = np.array([0.001755744, 0.001663332, 0.001677678])
        # args.mean_d = np.array([0.0, 0.0, 0.0])
        # args.std_d = np.array([1.0, 1.0, 1.0])
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
    args.dataf = f'data/data_{args.data_type}'

    # n_his
    args.outf += '_nHis%d' % args.n_his
    # args.evalf += '_nHis%d' % args.n_his

    # data augmentation
    if args.augment_ratio > 0:
        args.outf += '_aug%.2f' % args.augment_ratio
        # args.evalf += '_aug%.2f' % args.augment_ratio

    args.outf += f'_gt{args.gt_particles}'
    args.outf += f'_seqlen{args.sequence_length}'
    
    # args.outf += f'_{args.loss_type}'
    if args.loss_type == 'l1shape':
        args.outf += f'_l1shape'
    else:
        args.outf += f'_emd{args.emd_weight}'
        args.outf += f'_chamfer{args.chamfer_weight}'
        args.outf += f'_uh{args.uh_weight}'
        args.outf += f'_clip{args.clip_weight}'

    # evaluation checkpoints
    if args.stage in ['dy']:
        if args.eval_epoch > -1:
            args.evalf += '_Epoch_' + str(args.eval_epoch)
            args.evalf += '_Iter_' + str(args.eval_iter)
        else:
            args.evalf += '_Epoch_best'

        args.evalf += '_%s' % args.eval_set




    return args
