#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out


# export PYTHONPATH="/viscam/u/hshi74/projects/deformable/baselines:/viscam/u/hshi74/projects/deformable/PlasticineLab"
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior
# export LD_LIBRARY_PATH="/sailhome/hshi74/my_lib/:$LD_LIBRARY_PATH"


# Task 1: N Grip
kernprof -l train.py \
	--env Gripper \
	--data_type ngrip_fixed \
	--stage dy \
	--gen_data 0 \
	--gen_stat 0 \
	--gen_vision 0 \
	--num_workers 4 \
	--resume 0 \
	--resume_epoch 0 \
	--resume_iter 0 \
	--lr 0.0001 \
	--optimizer Adam \
	--batch_size 4 \
	--n_his 4 \
	--verbose_data 0 \
	--verbose_model 0 \
	--log_per_iter 100 \
	--valid 0 \
	--stdreg 0 \
	--matched_motion 0 \
	--matched_motion_weight 0.0 \
	--eval 0 \
	--gt_particles 0 \
    --shape_aug 1 \
	--n_epoch 100 \
	--n_rollout 50 \
	--ckp_per_iter 10000 \
	--losstype $1 \
	--sequence_length $2 \
	--emd_weight $3 \
	--chamfer_weight $4 \
	--uh_weight $5 \
	--clip_weight $6 \
	--augment_ratio $7 \
	--p_rigid $8


# Task 2: N Grip 3D
# kernprof -l train.py \
# 	--env Gripper \
# 	--data_type ngrip_3d \
# 	--stage dy \
# 	--gen_data 0 \
# 	--gen_stat 0 \
# 	--gen_vision 0 \
# 	--num_workers 4 \
# 	--resume 0 \
# 	--resume_epoch 0 \
# 	--resume_iter 0 \
# 	--lr 0.0001 \
# 	--optimizer Adam \
# 	--batch_size 4 \
# 	--n_his 4 \
# 	--verbose_data 0 \
# 	--verbose_model 0 \
# 	--log_per_iter 100 \
# 	--valid 0 \
# 	--stdreg 0 \
# 	--matched_motion 0 \
# 	--matched_motion_weight 0.0 \
# 	--eval 0 \
# 	--gt_particles 0 \
# 	--shape_aug 1 \
# 	--n_epoch 100 \
# 	--n_rollout 90 \
# 	--ckp_per_iter 10000 \
# 	--losstype $1 \
# 	--sequence_length $2 \
# 	--emd_weight $3 \
# 	--chamfer_weight $4 \
# 	--uh_weight $5 \
# 	--clip_weight $6 \
# 	--augment_ratio $7 \
# 	--p_rigid $8
