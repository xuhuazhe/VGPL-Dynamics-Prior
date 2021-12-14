#!/usr/bin/env bash
# cd /viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior
# cd /viscam/u/hshi74/projects/deformable/VGPL-Dynamics-Prior

# declare parameters
declare -a arr=("predict") # control_algo: fix, search, *predict
declare -a arr2=("5") # n_grips: 2,*3,4,5,6
declare -a arr3=("GD") # opt_algo: CEM, *GD, CEM_GD
declare -a arr4=("1") # correction: 0, *1
declare -a arr5=("emd") # reward_type: *emd, chamfer, emd_chamfer_uh_clip
declare -a arr6=("fish" "clover" "heart" "flower" "moon" "controller" "hat" "nut" "butterfly") 
declare -a arr7=("0") # debug: 0, 1
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
                            # for p in "${arr8[@]}"
                            # do
                            echo "$i $j $k $l $m $n $o"
                            sbatch ./scripts/control/control.sh $i $j $k $l $m $n $o
                            # sbatch ./scripts/control/control_render.sh $i $j $k $l $m $n $o
                            
                            # bash ./scripts/control/control.sh $i $j $k $l $m $n $o
                            # bash ./scripts/control/control_render.sh $i $j $k $l $m $n $o
                            # done
                        done
                    done
                done
            done
        done
    done
done
