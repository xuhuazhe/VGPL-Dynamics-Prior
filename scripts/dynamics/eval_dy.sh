#!/usr/bin/env bash

# Task 1: N Grip
python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vis plt \
	--data_type ngrip \
	--n_rollout 10 \
	--outf_eval dump/dump_ngrip/files_dy_01-Dec-2021-11:43:34.102959_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 95 \
	--eval_iter 225 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1

# Task 2: N Grip 3D
# python eval.py \
# 	--env Gripper \
# 	--stage dy \
# 	--eval_set train \
# 	--verbose_data 0 \
# 	--n_his 4 \
# 	--augment 0.05 \
# 	--vis plt \
# 	--data_type n_grip_3d \
# 	--n_rollout 10 \
# 	--outf_eval dump/dump_ngrip_3d/files_dy_02-Dec-2021-00:34:15.103354_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
# 	--eval_epoch 99 \
# 	--eval_iter 1601 \
# 	--sequence_length 6 \
# 	--gt_particles 0 \
# 	--shape_aug 1