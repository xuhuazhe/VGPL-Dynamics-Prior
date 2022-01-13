#!/usr/bin/env bash

python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vis plt \
    --data_type ngrip_fixed_robot_v1 \
	--n_rollout 10 \
    --outf_eval dump/dump_ngrip_fixed_robot_v1/files_dy_12-Jan-2022-17:11:14.314176_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 0 \
	--eval_iter 10 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1

