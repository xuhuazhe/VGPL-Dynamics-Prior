#!/usr/bin/env bash

python eval.py \
	--env Gripper \
	--stage dy \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment 0.05 \
	--vis plt \
    --data_type ngrip_fixed_v3 \
	--n_rollout 10 \
    --outf_eval dump/dump_ngrip_fixed_v3/files_dy_21-Dec-2021-10:58:53.121924_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--eval_epoch 99 \
	--eval_iter 1409 \
	--sequence_length 6 \
	--gt_particles 0 \
	--shape_aug 1

