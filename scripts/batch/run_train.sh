#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("emd_uh_clip")
declare -a arr2=("5")
declare -a arr3=("0.05")
declare -a arr4=("0.0")
for i in "${arr[@]}"
do
    for j in "${arr2[@]}"
    do
        for k in "${arr3[@]}"
        do
            for w in "${arr4[@]}"
            do
                echo "$i $j $k $w"
                sbatch ./scripts/dynamics/train_dy.sh $i $j $k $w
                # bash ./scripts/dynamics/train_dy.sh $i $j $k $w
            done
	    done
    done
done
