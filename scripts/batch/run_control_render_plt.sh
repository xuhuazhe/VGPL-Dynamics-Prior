#!/usr/bin/env bash

# declare parameters
declare -a arr=("E" "R" "S")
for i in "${arr[@]}"
do
    echo "$i"
    bash ./scripts/control/control_render_plt.sh $i "alphabet_bold"
done