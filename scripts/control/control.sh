#!/usr/bin/env bash
kernprof -l control.py \
	--env Gripper \
	--stage dy \
	--outf_control dump/dump_ngrip/files_dy_25-Oct-2021-15:09:15.587966_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 95 \
	--eval_iter 225 \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05
