import os
import time
import cv2

import numpy as np
import torch
import torch.nn.functional as F

from config import gen_args
from data_utils import load_data, get_scene_info, normalize_scene_param
from data_utils import get_env_group, prepare_input, denormalize
from models import Model
from utils import add_log, convert_groups_to_colors
from utils import create_instance_colors, set_seed, Tee, count_parameters


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.evalf)
os.system('mkdir -p ' + os.path.join(args.evalf, 'render'))

tee = Tee(os.path.join(args.evalf, 'eval.log'), 'w')


### evaluating

data_names = args.data_names

use_gpu = 0 #torch.cuda.is_available()

infos = np.arange(10)

for idx_episode in range(len(infos)):

    print("Rollout %d / %d" % (idx_episode, len(infos)))

    B = 1
    n_particle, n_shape = 0, 0

    # ground truth
    datas = []
    p_gt = []
    s_gt = []
    for step in range(args.time_step):
        data_path = os.path.join(args.dataf, 'train', str(infos[idx_episode]).zfill(3), str(step) + '.h5')

        data = load_data(data_names, data_path)
        if n_particle == 0 and n_shape == 0:
            n_particle, n_shape, scene_params = get_scene_info(data)
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
            print(state_cur[:, -1, :])
            # attr: (n_p + n_s) x attr_dim
            # Rr_cur, Rs_cur: n_rel x (n_p + n_s)
            # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
            attr, _, Rr_cur, Rs_cur = prepare_input(state_cur[-1].cpu().numpy(), n_particle, n_shape, args)

            st_time = time.time()

            # unsqueeze the batch dimension
            # attr: B x (n_p + n_s) x attr_dim
            # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            attr = attr.unsqueeze(0)
            Rr_cur = Rr_cur.unsqueeze(0)
            Rs_cur = Rs_cur.unsqueeze(0)
            state_cur = state_cur.unsqueeze(0)

