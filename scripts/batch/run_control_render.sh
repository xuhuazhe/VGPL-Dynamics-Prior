#!/usr/bin/env bash

# declare parameters
declare -a arr=("B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" "Z")
for i in "${arr[@]}"
do
    echo "$i"
    sbatch ./scripts/control/control.sh $i
done