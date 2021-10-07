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
	--outf_eval dump/dump_ngrip/files_dy_06-Oct-2021-00:34:48.734371_nHis4_aug0.05_gt0_chamfer_uh_seqlen5_uhw0.02_clipw0.0 \
	--eval_epoch 94 \
	--eval_iter 42
