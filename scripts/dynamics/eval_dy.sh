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
	--outf_eval dump/dump_ngrip/files_dy_24-Oct-2021-10:35:24.617190_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.15_clip0.0 \
	--eval_epoch 95 \
	--eval_iter 225 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1
