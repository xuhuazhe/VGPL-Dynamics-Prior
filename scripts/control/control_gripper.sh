#!/usr/bin/env bash
python control.py 		\
	--env Gripper 		\
	--stage dy 		\
	--eval_epoch 517    \
	--eval_iter 254      \
	--eval_set train 	\
	--verbose_data 0	\
	--sequence_length 49	\
	--n_his 4		\
	--augment 0.05
