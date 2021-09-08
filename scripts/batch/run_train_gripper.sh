#!/usr/bin/env bash
cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("emd_uh")
declare -a arr2=("10")
declare -a arr3=("0.1" "0.05" "0.02")
for i in "${arr[@]}"
do
    for j in "${arr2[@]}"
    do
        for k in "${arr3[@]}"
        do
            echo "$i"
	    echo "$j"
	    echo "$k"
            sbatch ./scripts/batch/gripper_train.sh $i $j $k
	done
    done
done
