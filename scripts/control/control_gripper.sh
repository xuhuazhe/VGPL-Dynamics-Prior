#!/usr/bin/env bash
python control.py 		\
	--env Gripper 		\
	--stage dy 		\
	--eval_epoch 92    \
	--eval_iter 472      \
	--eval_set train 	\
	--verbose_data 0	\
	--n_frames 49	\
	--n_his 4		\
	--augment 0.05
