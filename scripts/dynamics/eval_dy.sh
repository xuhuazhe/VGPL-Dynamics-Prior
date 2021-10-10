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
	--outf_eval dump/dump_ngrip/files_dy_06-Oct-2021-19:18:45.700479_nHis4_aug0.05_gt1_chamfer_uh_clip_seqlen7_uhw0.05_clipw0.0 \
	--eval_epoch 96 \
	--eval_iter 336
