#!/usr/bin/env bash

python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vis plt \
    --data_type ngrip_3d_fixed \
	--n_rollout 10 \
    --outf_eval dump/dump_ngrip_3d_fixed/files_dy_18-Dec-2021-23:37:56.024302_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 93 \
	--eval_iter 681 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1

