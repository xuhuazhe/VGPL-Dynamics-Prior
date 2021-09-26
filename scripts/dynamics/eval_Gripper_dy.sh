#!/usr/bin/env bash
python eval_dy_gripper.py 		\
	--env Gripper 		\
	--stage dy 		\
	--eval_epoch 97    \
	--eval_iter 186      \
	--eval_set train 	\
	--verbose_data 0	\
	--n_frames 49	\
	--n_his 4		\
	--augment 0.05		\
	--vispy 1
