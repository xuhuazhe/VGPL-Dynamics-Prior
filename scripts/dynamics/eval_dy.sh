#!/usr/bin/env bash
python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vis plt \
	--data_type ngrip_3d \
	--n_rollout 1 \
	--outf_eval dump/dump_ngrip_3d/files_dy_02-Dec-2021-00:34:15.103354_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 99 \
	--eval_iter 1601 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1
