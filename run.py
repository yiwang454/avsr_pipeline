import click
import glob
from video import load_video
from scene import segment_scenes
from facetrack import load_face_detector, find_facetracks
from syncnet import load_syncnet, find_talking_segments
import pandas as pd
import os

@click.command(context_settings=dict(show_default=True))
@click.option('--device', default='cuda:0', help='CUDA device.')
@click.option('--scene-threshold', default=0.004, help='Threshold for histogram based shot detection.')
@click.option('--min-scene-duration', default=25, help='Minimum scene duration in frames.')
@click.option('--min-face-size', default=50, help='Minimum mean face size in pixels.')
@click.option('--detect-face-every-nth-frame', default=1, help='Detect faces every nth frames.')
@click.option('--syncnet-threshold', default=2.5, help='SyncNet threshold.')
@click.option('--min-speech-duration', default=20, help='Minimum speech segment duration.')
@click.option('--max-pause-duration', default=10, help='Maximum pause duration between speech segments.')
@click.argument('pattern')
@click.argument('output_dir')
def main(device, scene_threshold, min_scene_duration, min_face_size, detect_face_every_nth_frame, syncnet_threshold, min_speech_duration, max_pause_duration, pattern, output_dir):
    face_detector = load_face_detector(device)
    syncnet = load_syncnet(device)

    path_seg_info = dict()

    for video_dir in os.listdir(pattern):
        for video_file in os.listdir(os.path.join(pattern, video_dir)):
            path = os.path.join(pattern, video_dir, video_file)
            if path.endswith("mp4"):

                name = path.split('/')[-1].rsplit('.', 1)[0]
                print("Processing %s" % name)

                audio_path = path.replace('.mp4', '.wav').replace('video', 'audio')
                video = load_video(path, audio_path)
                scenes = segment_scenes(video, scene_threshold, min_scene_duration)
                effective_time = 0
                pieces = 0

                for scene in scenes:
                    scene = scene.trim()
                    if (len(scene.frames) == 0):
                       continue

                    facetracks = find_facetracks(face_detector, scene, min_face_size, detect_face_every_nth_frame)
                    for facetrack in facetracks:
                        segments = find_talking_segments(syncnet, facetrack, syncnet_threshold, min_speech_duration, max_pause_duration)
                        pieces += len(segments)
                        for segment in segments:
                            start = segment.frame_offset / 25.
                            end = start + len(segment.frames) / 25.
                            effective_time += len(segment.frames) / 25.

                            # segment.write('%s/%s-%.2f-%.2f.mp4' % (output_dir, name, start, end))
                path_seg_info["path"] = path
                path_seg_info["effective_time"] = effective_time
                path_seg_info["pieces"] = pieces

    df_seg_info = pd.DataFrame(path_seg_info)
    df_seg_info.to_csv(output_dir, index=False)


if __name__ == '__main__':
    main()
