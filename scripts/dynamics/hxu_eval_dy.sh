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
	--outf_eval dump/dump_ngrip/tmp \
	--eval_epoch 94 \
	--eval_iter 42 \
	--sequence_length 5 \
	--gt_particles 0 \
	--shape_aug 1
