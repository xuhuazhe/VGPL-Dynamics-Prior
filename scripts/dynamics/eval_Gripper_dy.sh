#!/usr/bin/env bash
python eval_dy_gripper.py 		\
	--env Gripper 		\
	--stage dy 		\
	--eval_epoch 159    \
	--eval_iter 77      \
	--eval_set train 	\
	--verbose_data 0	\
	--n_frames 59    \
	--n_his 4		\
	--augment 0.05		\
	--vispy 1
