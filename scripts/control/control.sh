#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out

kernprof -l control.py \
	--env Gripper \
	--data_type ngrip_fixed_small \
	--stage control \
	--outf_control dump/dump_ngrip_fixed_small/files_dy_22-Dec-2021-10:47:36.944946_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--gripperf ../PlasticineLab/plb/envs/gripper_fixed.yml \
	--eval_epoch 99 \
	--eval_iter 1409 \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment_ratio 0.05 \
	--shape_aug 1 \
	--use_sim 0 \
	--gt_action 0 \
	--gt_state_goal 0 \
	--subgoal 0 \
	--control_sample_size 500 \
	--control_batch_size 4 \
	--predict_horizon 2 \
	--CEM_opt_iter 3 \
	--CEM_init_pose_sample_size 80 \
	--CEM_gripper_rate_sample_size 4 \
	--GD_batch_size 1 \
	--control_algo $1 \
	--n_grips $2 \
	--opt_algo $3 \
	--correction $4 \
	--reward_type $5 \
	--goal_shape_name $6 \
	--debug $7
