#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Pinch
#SBATCH --partition=viscam
#SBATCH --nodelist=viscam2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/sailhome/hxu/VGPL/%A.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=20
source ~/.bashrc
conda activate deformable
export PYTHONPATH="/viscam/u/hxu/projects/deformable/baselines:/viscam/u/hxu/projects/deformable/PlasticineLab"
cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
export LD_LIBRARY_PATH="/sailhome/hxu/my_lib/:$LD_LIBRARY_PATH"

CUDA_VISIBLE_DEVICES=0		\
python train.py 		\
	--env Pinch 		\
	--stage dy		\
	--gen_data 0 		\
	--gen_stat 0		\
	--gen_vision 0		\
	--num_workers 20 	\
	--resume 0		\
	--resume_epoch 0	\
	--resume_iter 0		\
	--lr 0.0001		\
	--optimizer Adam	\
	--batch_size 4		\
	--n_his 4		\
	--sequence_length 5	\
	--augment 0.05		\
	--verbose_data 0 	\
	--verbose_model 0	\
	--log_per_iter 100	\
	--ckp_per_iter 5000	\
	--eval 0            \
	--losstype emd      \
	--stdreg  1         \
        --stdreg_weight $1
