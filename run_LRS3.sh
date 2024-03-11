  for i in 1; do
    python3 run.py --device cuda:$((i-1)) $resampled_dir/input_files.0${i} $output_dir | tee $output_dir/log.$i &
  done
