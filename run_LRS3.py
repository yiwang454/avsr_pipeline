import subprocess
import os

GPU_nums = [0, 0, 0, 0]
ss_num = len(GPU_nums)
GPU_idx = 0

resampled_dir = "/mnt/ceph_rbd/muavic_project/muavic/fr/video/train"
output_file = "/mnt/ceph_rbd/muavic_project/avsr_pipeline/segments.csv"

COMMAND_LIST = []
for i in 1; do
python3
run.py - -device
cuda:$((i - 1)) $resampled_dir $output_file
done
