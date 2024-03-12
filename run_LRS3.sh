  resampled_dir="/mnt/ceph_rbd/muavic_project/muavic/fr/video/valid_fake"
  output_file="/mnt/ceph_rbd/muavic_project/avsr_pipeline/segments.csv"
  cd /mnt/ceph_rbd/muavic_project/avsr_pipeline
  for i in 1; do
    python3 run.py --device cuda:$((i-1)) $resampled_dir $output_file
  done
