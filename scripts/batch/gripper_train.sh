#!/usr/bin/env bash
#SBATCH --job-name=VGPL-Gripper
#SBATCH --partition=viscam
#SBATCH --nodelist=viscam1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=/sailhome/hxu/VGPL/%A.out
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=20
source ~/.bashrc
conda activate deformable
export PYTHONPATH="/viscam/u/hxu/projects/deformable/baselines:/viscam/u/hxu/projects/deformable/PlasticineLab"
cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
export LD_LIBRARY_PATH="/sailhome/hxu/my_lib/:$LD_LIBRARY_PATH"

declare -a arr=("emd")
declare -a arr2=("10" "15" "20")
for i in "${arr[@]}"
do
    for j in "${arr2[@]}"
    do
        echo "$i"
	echo "$j"
        bash scripts/dynamics/train_Gripper_dy.sh $i $j
    done
done
