#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("emd_uh_clip")
declare -a arr2=("5")
declare -a arr3=("0.05")
declare -a arr4=("0.0")
declare -a arr5=("0.05")
for i in "${arr[@]}"
do
    for j in "${arr2[@]}"
    do
        for k in "${arr3[@]}"
        do
            for l in "${arr4[@]}"
            do
                for m in "${arr5[@]}"
                do
                    echo "$i $j $k $l $m"
                    sbatch ./scripts/dynamics/train_dy.sh $i $j $k $l $m
                    # bash ./scripts/dynamics/train_dy.sh $i $j $k $l $m
                done
            done
	    done
    done
done
