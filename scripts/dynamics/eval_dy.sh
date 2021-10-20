#!/usr/bin/env bash
python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vispy 1 \
	--data_type ngrip \
	--n_rollout 50 \
	--outf_eval dump/dump_ngrip/files_dy_18-Oct-2021-23:12:01.732749_nHis4_aug0.05_gt0_emd_uh_clip_seqlen5_uhw0.0_clipw0.0 \
	--eval_epoch 94 \
	--eval_iter 42 \
	--sequence_length 5 \
	--gt_particles 0 \
	--shape_aug 1
