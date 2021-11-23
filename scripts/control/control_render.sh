#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out

python control_render.py \
	--env Gripper \
	--data_type ngrip \
	--stage dy \
	--outf_control dump/dump_ngrip/files_dy_25-Oct-2021-15:09:15.587966_nHis4_aug0.05_gt0_seqlen6_emd0.3_chamfer0.7_uh0.1_clip0.0 \
	--gripperf ../PlasticineLab/plb/envs/gripper.yml \
	--shape_aug 1 \
	--opt_algo GD \
	--n_grips 3 \
	--opt_iter 3 \
	--sample_iter 3 \
	--rewardtype emd \
	--use_sim 0 \
	--gt_action 0 \
	--gt_state_goal 0 \
	--debug 0