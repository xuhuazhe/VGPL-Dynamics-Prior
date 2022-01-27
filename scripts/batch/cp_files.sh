cd /scr/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/sim_control_final

# "A" "B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" "Z"
declare -a arr=("A" "B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" "Z")
for i in "${arr[@]}"
do
    # mkdir -p $i/selected_tool
    # mv $i/* $i/selected/
    # mkdir -p $i
    # sshpass -p "Hataraki213300?!" scp hshi74@scdt.stanford.edu:/viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/sim_control_final/$i/selected_tool/act_seq_opt.npy $i/selected
    # sshpass -p "Hataraki213300?!" scp hshi74@scdt.stanford.edu:/viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/sim_control_final/$i/selected_tool/init_pose_seq_opt.npy $i/selected_tool
    sshpass -p "Hataraki213300?!" scp hshi74@scdt.stanford.edu:/viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/sim_control_final/$i/selected/control.log $i/
    # sshpass -p "Hataraki213300?!" scp hshi74@scdt.stanford.edu:/viscam/u/hxu/projects/deformable/VGPL-Dynamics-Prior/dump/sim_control_final/$i/selected/goal_particles.png $i/
done