#!/usr/bin/env bash
python eval_dy_pinch.py 		\
	--env Gripper 		\
	--stage dy 		\
	--eval_epoch 78    \
	--eval_iter 188      \
	--eval_set train 	\
	--verbose_data 0	\
	--sequence_length 49	\
	--n_his 4		\
	--augment 0.05		\
	--vispy 1
