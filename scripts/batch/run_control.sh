#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

# currently a placeholder
declare -a arr=("emd") # rewardtype: emd, l1shape
declare -a arr2=("1") # use_sim: 0, 1
declare -a arr2=("0") # gt_action: 0, 1
declare -a arr3=("1") # gt_state_goal: 0, 1
# declare -a arr4=("0.7") # use_sim: 0, 1
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
