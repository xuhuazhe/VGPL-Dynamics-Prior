#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=viscam-interactive
#SBATCH --nodelist=viscam1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/sailhome/hshi74/output/deformable/%A.out
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=10
# source ~/.bashrc
# conda activate deformable
# export PYTHONPATH="/viscam/u/hshi74/projects/deformable/baselines:/viscam/u/hshi74/projects/deformable/PlasticineLab"
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior
# export LD_LIBRARY_PATH="/sailhome/hshi74/my_lib/:$LD_LIBRARY_PATH"


bash scripts/dynamics/train_Gripper_dy.sh $1 $2 $3 $4
