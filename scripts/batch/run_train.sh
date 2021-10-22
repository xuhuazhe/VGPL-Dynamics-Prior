#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("L2Shape")
declare -a arr2=("5")
declare -a arr3=("0.0")
declare -a arr4=("0.0")
declare -a arr5=("0.0")
declare -a arr6=("0.05")
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
                    for n in "${arr6[@]}"
                    do
                        echo "$i $j $k $l $m $n"
                        sbatch ./scripts/dynamics/train_dy.sh $i $j $k $l $m $n
                        # bash ./scripts/dynamics/train_dy.sh $i $j $k $l $m $n
                    done    
                done
            done
	    done
    done
done
