import subprocess
import os

# GPU_nums = [0]
# ss_num = len(GPU_nums)
GPU_idx = 0

resampled_dir = "/mnt/ceph_rbd/muavic_project/muavic/fr/video/train"
output_file = "/mnt/ceph_rbd/muavic_project/avsr_pipeline/segments.csv"
root_dir = "/mnt/ceph_rbd/muavic_project/avsr_pipeline"
command = "python3 run_multiprocess.py --device cuda:{} {} {}".format(GPU_idx, resampled_dir, output_file)
subp = subprocess.Popen(command, shell=True, cwd=root_dir,
                        encoding="utf-8")  # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
subp.wait()
if subp.poll() == 0:
    print(subp_ss[q].communicate())
else:
    print(command, 'fail')