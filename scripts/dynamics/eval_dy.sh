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
	--data_type ngrip_smooth \
	--n_rollout 10 \
	--outf_eval dump/dump_ngrip_smooth/files_dy_08-Dec-2021-10:49:57.281936_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
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
# 	--data_type ngrip_3d \
# 	--n_rollout 10 \
# 	--outf_eval dump/dump_ngrip_3d/files_dy_05-Dec-2021-16:57:42.151450_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
# 	--eval_epoch 99 \
# 	--eval_iter 1601 \
# 	--sequence_length 6 \
# 	--gt_particles 0 \
# 	--shape_aug 1