#!/usr/bin/env bash
python eval_dy_gripper.py 		\
	--env Gripper 		\
	--stage dy 		\
	--eval_epoch 92    \
	--eval_iter 472      \
	--eval_set train 	\
	--verbose_data 0	\
	--n_his 4		\
	--augment 0.05		\
	--vispy 1
