#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import ffmpeg
import random
import torchvision
import torchmetrics
from tqdm.rich import tqdm

from .utils import read_video_frames_async, get_video_info
from .FaceBoxes import FaceBoxesDetector, FaceBoxesTracker

class TrackSplitEngine:
    def __init__(self, output_dir='./output', device='cuda', silent=False):
        random.seed(42)
        self._device = device
        self._silent = silent
        self.max_miss = 5
        self.min_face_size = 128
        self.min_track_length = 150
        # paths and data engine
        self.face_detector = FaceBoxesDetector('./assets/FaceBoxes.pth')
        self.face_detector = self.face_detector.to(self._device)
        self.face_detector.eval()
        # output dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'meta_data'), exist_ok=True)

    def run_video_pipeline(self, video_path):
        video_meta_data = get_video_info(video_path)
        video_name = video_path.split('/')[-1].split('.')[0]
        # run face detection
        if os.path.exists(f'{self.output_dir}/meta_data/{video_name}.pt'):
            meta_data = torch.load(f'{self.output_dir}/meta_data/{video_name}.pt', weights_only=False)
            all_face_data = meta_data['all_face_data']
            hard_trans_id = meta_data['hard_trans_id']
            tracks = meta_data['tracks']
            if not self._silent:
                print(f'Loaded face detection results from {video_name}...')
        else:
            if not self._silent:
                print(f'Detecting faces in {video_path}...')
            hard_trans_id, all_face_data, last_frame = [], [], None
            for frame_idx, frame in enumerate(tqdm(read_video_frames_async(video_path), total=video_meta_data['video']['num_frames'])):
                frame = frame.to(self._device)
                face_data = self.face_detector.detect(frame)
                if calc_ssim(frame, last_frame) < 0.6:
                    hard_trans_id.append(frame_idx)
                last_frame = frame
                if face_data is not None:
                    all_face_data.append({'frame': frame_idx, 'bboxes': face_data.cpu().numpy()})

            # run face tracking
            if not self._silent:
                print(f'Tracking faces in {video_name}...')
            face_tracker = FaceBoxesTracker(max_miss=self.max_miss, min_length=self.min_track_length, min_size=self.min_face_size)
            for face_data in all_face_data:
                face_tracker.update(face_data['bboxes'], face_data['frame'])
            tracks = face_tracker.get_tracked_faces()
            tracks = split_tracks_with_hard_trans(tracks, hard_trans_id, min_length=self.min_track_length)
            tracks = split_too_long_tracks(tracks, hope_length=25*120)
            for track in tracks:
                track['bbox'] = expand_bbox(track['bbox'], video_meta_data['video']['height'], video_meta_data['video']['width'], 1.8)
            meta_data = {'video_meta_data': video_meta_data, 'all_face_data': all_face_data, 'tracks': tracks, 'hard_trans_id': hard_trans_id}
            torch.save(meta_data, f'{self.output_dir}/meta_data/{video_name}.pt')
        
        # crop and save faces
        if not self._silent:
            print(f'Dumping tracked faces in {video_name}...')
        for track in tracks:
            start_time = track['start_frame'] / video_meta_data['video']['frame_rate']
            end_time = track['end_frame'] / video_meta_data['video']['frame_rate']
            left, top, width, height = track['bbox']
            output_path = f'{self.output_dir}/videos/{video_name}_S{track["start_frame"]}_E{track["end_frame"]}_L{left}_T{top}_W{width}_H{height}.mp4'
            if os.path.exists(output_path):
                continue
            try:
                video_stream = ffmpeg.input(video_path, ss=start_time, to=end_time).video.crop(left, top, width, height)
                audio_stream = ffmpeg.input(video_path, ss=start_time, to=end_time).audio
                ffmpeg.output(
                    video_stream, audio_stream, output_path,
                    r=25, vcodec='h264_nvenc', acodec='aac', ar=16000, strict='experimental' # copy or h264_nvenc or libx264
                ).run(overwrite_output=True, quiet=True)
            except Exception as e:
                print("Failed to crop video for {}, error: {}".format(output_path, e))
                continue
        if not self._silent:
            print('Done!')


def calc_ssim(frame, last_frame, data_range=(0, 255)):
    if last_frame is None:
        return 1.0
    # resize frame last frame to short size 600
    frame = torchvision.transforms.functional.resize(frame, 600)
    last_frame = torchvision.transforms.functional.resize(last_frame, 600)
    frame_gray = torchvision.transforms.functional.rgb_to_grayscale(frame[None]).float()
    last_frame_gray = torchvision.transforms.functional.rgb_to_grayscale(last_frame[None]).float()
    ssim = torchmetrics.functional.image.structural_similarity_index_measure(
        frame_gray, last_frame_gray, data_range=data_range,
    )
    return ssim.item()


def split_tracks_with_hard_trans(tracks, hard_trans_id, min_length=150):
    if not hard_trans_id:
        return tracks
    split_tracks = []
    for track in tracks:
        start_frame = track['start_frame']
        end_frame = track['end_frame']
        hard_trans_in_track = [ht for ht in hard_trans_id if start_frame < ht < end_frame]
        if not hard_trans_in_track:
            split_tracks.append(track)
        else:
            hard_trans_in_track.sort()
            split_points = [start_frame] + hard_trans_in_track + [end_frame]
            for i in range(len(split_points) - 1):
                segment_start = split_points[i]
                segment_end = split_points[i + 1]
                new_track = track.copy()
                new_track['start_frame'] = segment_start
                new_track['end_frame'] = segment_end
                if new_track['end_frame'] - new_track['start_frame'] < min_length:
                    continue
                split_tracks.append(new_track)
    return split_tracks


def split_too_long_tracks(tracks, hope_length=25*150):
    split_tracks = []
    for track in tracks:
        start_frame = track['start_frame']
        end_frame = track['end_frame']
        track_length = end_frame - start_frame
        if track_length <= hope_length:
            split_tracks.append(track)
        else:
            track_segments = []
            current_start = start_frame
            while current_start < end_frame:
                current_end = min(current_start + hope_length, end_frame)
                new_track = track.copy()
                new_track['start_frame'] = current_start
                new_track['end_frame'] = current_end
                track_segments.append(new_track)
                current_start = current_end
            if len(track_segments) > 1:
                last_segment = track_segments[-1]
                second_last_segment = track_segments[-2]
                last_length = last_segment['end_frame'] - last_segment['start_frame']
                if last_length < hope_length // 2:
                    second_last_segment['end_frame'] = last_segment['end_frame']
                    track_segments.pop()
            split_tracks.extend(track_segments)
    return split_tracks


def expand_bbox(bbox, image_height, image_width, bbox_scale=1.42):
    x1, y1, w, h = bbox
    center_x, center_y = x1 + w // 2, y1 + h // 2
    size = int(max(w, h) * bbox_scale)
    new_x1_min, new_y1_min = int(center_x - size // 2), int(center_y - size // 2)
    new_x1_max, new_y1_max = int(center_x + size // 2), int(center_y + size // 2)
    if new_x1_min < 0 or new_y1_min < 0:
        min_overflow = min(new_x1_min, new_y1_min)
        new_x1_min += -min_overflow
        new_y1_min += -min_overflow
    if new_x1_max > image_width - 1 or new_y1_max > image_height - 1:
        max_overflow = max(new_x1_max - image_width - 1, new_y1_max - image_height - 1)
        new_x1_max -= max_overflow
        new_y1_max -= max_overflow
    new_bbox = [new_x1_min, new_y1_min, new_x1_max - new_x1_min, new_y1_max - new_y1_min]
    return new_bbox

