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
	--outf_eval dump/dump_ngrip/files_dy_29-Oct-2021-14:21:58.477253_nHis4_aug0.05_gt0_seqlen5_l2shape \
	--eval_epoch 83 \
	--eval_iter 569 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1
