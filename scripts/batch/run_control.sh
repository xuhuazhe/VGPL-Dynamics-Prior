#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

# currently a placeholder
declare -a arr=("5") # n_grips: 1-6
declare -a arr2=("GD") # opt_algo: CEM, GD, CEM_GD
declare -a arr3=("3") # CEM_opt_iter
declare -a arr4=("0") # subgoal: 0, 1
declare -a arr5=("1") # correction: 0, 1
declare -a arr6=("emd") # reward_type: emd, chamfer
# goal_shape_name: vid_0-49, I, T "butterfly" "car" "elephant" "fish" "flower" "heart" "house" "panda" "star"
declare -a arr7=("vid_0" "vid_8" "I" "T" "butterfly" "car" "elephant" "fish" "flower" "heart" "house" "panda" "star") 
declare -a arr8=("0") # debug: 0, 1
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
                            for p in "${arr8[@]}"
                            do
                                echo "$i $j $k $l $m $n $o $p"
                                sbatch ./scripts/control/control.sh $i $j $k $l $m $n $o $p
                                # sbatch ./scripts/control/control_render.sh $i $j $k $l $m $n $o $p
                                # bash ./scripts/control/control.sh $i $j $k $l $m $n $o $p
                                # bash ./scripts/control/control_render.sh $i $j $k $l $m $n $o $p
                            done
                        done
                    done
                done
            done
        done
    done
done
