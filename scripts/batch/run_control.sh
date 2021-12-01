#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

# currently a placeholder
declare -a arr=("GD") # opt_algo: CEM, GD, CEM_GD
declare -a arr2=("3") # CEM_opt_iter
declare -a arr3=("0") # subgoal: 0, 1
declare -a arr4=("emd") # reward_type: emd
declare -a arr5=("butterfly" "car" "elephant" "fish" "flower" "heart" "house" "panda" "star") # goal_shape_name: vid_0-49, I, T
declare -a arr6=("0") # debug: 0, 1
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
                        sbatch ./scripts/control/control.sh $i $j $k $l $m $n
                        # sbatch ./scripts/control/control_render.sh $i $j $k $l $m $n
                        # bash ./scripts/control/control.sh $i $j $k $l $m $n
                        # bash ./scripts/control/control_render.sh $i $j $k $l $m $n
                    done
                done
            done
        done
    done
done
