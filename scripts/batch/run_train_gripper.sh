#!/usr/bin/env bash
cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("emd")
declare -a arr2=("10" "15" "20")
for i in "${arr[@]}"
do
    for j in "${arr2[@]}"
    do
        echo "$i"
	    echo "$j"
        sbatch ./scripts/batch/gripper_train.sh $i $j
    done
done
