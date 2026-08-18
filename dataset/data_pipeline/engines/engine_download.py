#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import ffmpeg
from .utils import download_raw_video

class DownloadEngine:
    def __init__(self, output_dir='./output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'raw_videos'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'merged_videos'), exist_ok=True)
        self.raw_root = os.path.join(self.output_dir, 'raw_videos')
        self.merge_root = os.path.join(self.output_dir, 'merged_videos')

    def run_download(self, youtube_video_id):
        if exists_video(self.raw_root, youtube_video_id):
            return
        download_success = download_raw_video(self.raw_root, youtube_video_id)
        if not download_success:
            print("Failed to download video: {}".format(youtube_video_id))
        return download_success

    def run_merge_raw_video(self, youtube_video_id, remove_raw=False):
        if not exists_video(self.raw_root, youtube_video_id):
            return
        this_raw_video_path = os.path.join(self.raw_root, "{}_video.mp4".format(youtube_video_id))
        this_raw_audio_path = os.path.join(self.raw_root, "{}_audio.mp4".format(youtube_video_id))
        this_video_path = os.path.join(self.merge_root, "{}.mp4".format(youtube_video_id))
        try:
            input_video = ffmpeg.input(this_raw_video_path).video
            input_audio = ffmpeg.input(this_raw_audio_path).audio
            ffmpeg.output(
                input_video, input_audio, this_video_path, 
                r=25, vcodec='h264_nvenc', acodec='aac', ar=16000, strict='experimental' # copy or h264_nvenc or libx264
            ).run(overwrite_output=True, quiet=True)
        except Exception as e:
            print("Failed to merge video for {}, error: {}".format(youtube_video_id, e))
            return
        if remove_raw:
            os.remove(this_raw_video_path)
            os.remove(this_raw_audio_path)


def exists_video(root_dir, video_id):
    this_raw_video_path = os.path.join(root_dir, "{}_video.mp4".format(video_id))
    this_raw_audio_path = os.path.join(root_dir, "{}_audio.mp4".format(video_id))
    return os.path.exists(this_raw_video_path) and os.path.exists(this_raw_audio_path)
