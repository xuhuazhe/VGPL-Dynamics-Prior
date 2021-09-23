#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
cd /home/haochen/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("emd_uh_clip")
declare -a arr2=("10")
declare -a arr3=("0.1")
declare -a arr4=("0.1" "0.5")
for i in "${arr[@]}"
do
    for j in "${arr2[@]}"
    do
        for k in "${arr3[@]}"
        do
            for w in "${arr4[@]}"
            do
                echo "$i"
                echo "$j"
                echo "$k"
                sbatch ./scripts/batch/gripper_train.sh $i $j $k $w
            done
	    done
    done
done
