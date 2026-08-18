import os
import ffmpeg
from pytubefix import YouTube

def download_raw_video(raw_root, video_id):
    this_url = 'https://www.youtube.com/watch?v={}'.format(video_id)
    yt = YouTube(this_url, use_oauth=True)
    try:
        video_stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        if video_stream.fps < 24:
            print("Failed to download video for {}, fps {}.".format(video_id, video_stream.fps))
            return False
        video_stream.download(output_path=raw_root, filename="{}_video.mp4".format(video_id))
        audio_stream.download(output_path=raw_root, filename="{}_audio.mp4".format(video_id))
    except Exception as e:
        print("Failed to download video for {}, error: {}".format(video_id, e))
        return False
    return True


def get_video_meta(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    height = int(video_stream['height'])
    width = int(video_stream['width'])
    if video_stream['r_frame_rate'] != video_stream['avg_frame_rate']:
        r_fps = float(video_stream['r_frame_rate'].split('/')[0]) / float(video_stream['r_frame_rate'].split('/')[1])
        avg_fps = float(video_stream['avg_frame_rate'].split('/')[0]) / float(video_stream['avg_frame_rate'].split('/')[1])
        if abs(r_fps - avg_fps) > 0.1:
            print("Warning {}: r_frame_rate={}, avg_frame_rate={}.".format(video_path, video_stream['r_frame_rate'], video_stream['avg_frame_rate']))
    fps = float(video_stream['r_frame_rate'].split('/')[0]) / float(video_stream['r_frame_rate'].split('/')[1])
    duration = float(video_stream['duration'])
    return height, width, fps, duration


def split_into_parts(input_list, split_total):
    all_parts_input_list = []
    for idx in range(split_total):
        all_parts_input_list.append(input_list[idx::split_total])
    return all_parts_input_list
