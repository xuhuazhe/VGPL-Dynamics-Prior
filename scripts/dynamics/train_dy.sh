#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=viscam
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16

# source ~/.bashrc
# conda activate deformable

# export PYTHONPATH="/viscam/u/hxu/projects/deformable/baselines:/viscam/u/hxu/projects/deformable/PlasticineLab"
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# export LD_LIBRARY_PATH="/sailhome/hxu/my_lib/:$LD_LIBRARY_PATH"

# export PYTHONPATH="/viscam/u/hshi74/projects/deformable/baselines:/viscam/u/hshi74/projects/deformable/PlasticineLab"
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior
# export LD_LIBRARY_PATH="/sailhome/hshi74/my_lib/:$LD_LIBRARY_PATH"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
	--env Gripper \
	--data_type ngrip \
	--stage dy \
	--gen_data 0 \
	--gen_stat 0 \
	--gen_vision 0 \
	--num_workers 1 \
	--resume 0 \
	--resume_epoch 0 \
	--resume_iter 0 \
	--lr 0.0001 \
	--optimizer Adam \
	--batch_size 4 \
	--n_his 4 \
	--augment 0.05 \
	--verbose_data 0 \
	--verbose_model 0 \
	--log_per_iter 100 \
	--valid 0 \
	--stdreg 0 \
	--matched_motion 0 \
	--matched_motion_weight 0.0 \
	--eval 1 \
	--gt_particles 1 \
	--n_epoch 100 \
	--n_rollout 50 \
	--ckp_per_iter 10000 \
	--losstype $1 \
	--sequence_length $2 \
	--uh_weight $3 \
	--clip_weight $4