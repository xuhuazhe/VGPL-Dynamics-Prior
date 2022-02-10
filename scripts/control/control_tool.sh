#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out

kernprof -l control_tool.py \
	--env Gripper \
	--data_type ngrip_fixed_small \
	--stage control \
	--outf_control  /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior/dump/dump_ngrip_fixed_v6/files_dy_18-Jan-2022-12:52:52.015211_nHis4_aug0.05_gt0_seqlen6_emd0.9_chamfer0.1_uh0.0_clip0.0/ \
	--outf_new /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior/dump/dump_ngrip_fixed_small_v3/files_dy_18-Jan-2022-14:12:27.294557_nHis4_aug0.05_gt0_seqlen6_emd0.9_chamfer0.1_uh0.0_clip0.0/ \
	--gripperf ../PlasticineLab/plb/envs/gripper_fixed.yml \
	--eval_epoch 93 \
	--eval_iter 681 \
	--eval_set train \
	--verbose_data 0 \
	--n_his 4 \
	--augment_ratio 0.05 \
	--shape_aug 1 \
	--use_sim 0 \
	--gt_action 0 \
	--gt_state_goal 0 \
	--subgoal 0 \
	--control_sample_size 200 \
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
	--shape_type $6 \
	--goal_shape_name $7 \
	--debug $8
