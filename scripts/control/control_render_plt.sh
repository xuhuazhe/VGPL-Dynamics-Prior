#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out

python control_render_plt.py \
	--env Gripper \
	--data_type ngrip_3d_fixed_v3 \
	--stage control \
	--outf_control dump/sim_control_final/$1/selected \
	--gripperf ../PlasticineLab/plb/envs/gripper_fixed.yml \
	--eval_epoch 95 \
	--eval_iter 225 \
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
	--goal_shape_name $1 \
	--shape_type $2