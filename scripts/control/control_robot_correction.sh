#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out

kernprof -l control_robot_correction.py \
	--env Gripper \
	--data_type ngrip_fixed_robot_v4 \
	--stage control \
	--outf_control dump/dump_ngrip_fixed_robot_v4/files_dy_21-Jan-2022-22:33:11.729243_nHis4_aug0.05_gt0_seqlen6_emd0.9_chamfer0.1_uh0.0_clip0.0 \
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
	--predict_horizon 1 \
	--CEM_opt_iter 3 \
	--CEM_init_pose_sample_size 40 \
	--CEM_gripper_rate_sample_size 4 \
	--GD_batch_size 1 \
	--control_algo $1 \
	--sample_method $2 \
	--n_grips $3 \
	--opt_algo $4 \
	--correction $5 \
	--reward_type $6 \
	--shape_type $7 \
	--goal_shape_name $8 \
	--debug $9
