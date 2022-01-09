#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=viscam
#SBATCH --nodelist=viscam3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out


# export PYTHONPATH="/viscam/u/hshi74/projects/deformable/baselines:/viscam/u/hshi74/projects/deformable/PlasticineLab"
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior
# export LD_LIBRARY_PATH="/sailhome/hshi74/my_lib/:$LD_LIBRARY_PATH"


# Task 1: N Grip
kernprof -l train.py \
	--env Gripper \
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
	--eval 1 \
	--gt_particles 0 \
    --shape_aug 1 \
	--n_epoch 100 \
	--n_rollout 90 \
	--ckp_per_iter 10000 \
	--sequence_length 6 \
	--emd_weight 0.3 \
	--chamfer_weight 0.7 \
	--uh_weight 0.1 \
	--clip_weight 0.0 \
	--augment_ratio 0.05 \
	--p_rigid 1.0 \
	--data_type $1 \
	--loss_type $2 \
	--alpha $3
