#!/usr/bin/env bash

python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vis plt \
   	--data_type ngrip_fixed_robot_v4 \
	--n_rollout 1 \
    	--outf_eval dump/dump_ngrip_fixed_robot_v4/files_dy_21-Jan-2022-22:33:11.729243_nHis4_aug0.05_gt0_seqlen6_emd0.9_chamfer0.1_uh0.0_clip0.0 \
	--eval_epoch 93 \
	--eval_iter 681 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1

