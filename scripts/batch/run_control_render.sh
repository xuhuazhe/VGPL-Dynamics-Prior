#!/usr/bin/env bash

# declare parameters
declare -a arr=("E" "F" "K" "L" "M" "N" "S" "W" "Z")
for i in "${arr[@]}"
do
    echo "$i"
    sbatch ./scripts/control/control_render.sh $i
done