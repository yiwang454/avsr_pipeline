  resampled_dir="muavic/it/video/train"
  output_file="/mnt/ceph_rbd/muavic_project/avsr_pipeline/segments.csv"
  for i in 1; do
    python3 avsr_pipeline/run.py --device cuda:$((i-1)) $resampled_dir $output_file
  done
