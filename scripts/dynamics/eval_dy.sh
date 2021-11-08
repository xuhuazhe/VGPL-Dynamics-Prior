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
	--outf_eval dump/dump_ngrip/files_dy_05-Nov-2021-11\:07\:07.526807_nHis4_aug0.05_gt0_seqlen5_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 63 \
	--eval_iter 465 \
	--sequence_length 5 \
	--gt_particles 0 \
	--shape_aug 1
