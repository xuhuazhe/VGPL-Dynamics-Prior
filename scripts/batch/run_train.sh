#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

declare -a arr=("emd_chamfer_uh_clip")
declare -a arr2=("6")
declare -a arr3=("0.3")
declare -a arr4=("0.7")
declare -a arr5=("0.1")
declare -a arr6=("0.0")
declare -a arr7=("0.05")
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
                        for o in "${arr7[@]}"
                        do
                            echo "$i $j $k $l $m $n $o"
                            sbatch ./scripts/dynamics/train_dy.sh $i $j $k $l $m $n $o
                            # bash ./scripts/dynamics/train_dy.sh $i $j $k $l $m $n $o
                        done
                    done    
                done
            done
	    done
    done
done
