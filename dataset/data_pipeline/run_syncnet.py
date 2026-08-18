#!/usr/bin/env python
# Copyright (c) Riocong Liu and Xuangeng Chu (xg.chu@outlook.com)

import os
import csv
import pickle
import shutil
import argparse
from tqdm import tqdm

from engines import SyncNetEngine
from engines.utils import split_into_parts, get_video_info

def find_synchronized_videos(output_dir, threshold):
    video_dir = os.path.join(output_dir, 'videos')
    syncnet_dir = os.path.join(output_dir, 'sync_data')
    pickle_files = os.listdir(syncnet_dir)
    pickle_files = [os.path.join(syncnet_dir, f) for f in pickle_files if f.endswith('.pkl')]
    qualified_files = []
    for file_path in tqdm(pickle_files):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            video_name, score = data['video_name'], data['conf']
            if score > threshold:
                video_info = get_video_info(os.path.join(video_dir, video_name+'.mp4'))
                video_length = video_info['video']['num_frames']
                qualified_files.append([video_name, score, video_length])
        except Exception as e:
            print(f"Reading file {file_path} failed: {e}")

    all_frames_length = 0
    with open(os.path.join(output_dir, 'metadata.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video_name', 'syncnet_score', 'video_length'])
        for video_name, score, video_length in qualified_files:
            writer.writerow([video_name, score, video_length])
            all_frames_length += video_length
    print(f"Total frames length: {all_frames_length/25.0/3600} hours")


def move_to_new_folder(output_dir):
    video_path = os.path.join(output_dir, 'videos')
    csv_file_path = os.path.join(output_dir, 'metadata.csv')
    sync_video_path = os.path.join(output_dir, 'videos_sync')
    os.makedirs(sync_video_path, exist_ok=True)
    all_video_names = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_video_names.append(row['video_name'])

    for video_name in tqdm(all_video_names):
        ori_path = os.path.join(video_path, video_name+'.mp4')
        tgt_path = os.path.join(sync_video_path, video_name+'.mp4')
        if os.path.exists(ori_path):
            try:
                shutil.move(ori_path, tgt_path)
            except Exception as e:
                print(f"Moving video {video_name} failed: {e}")
        else:
            print(f"Video {video_name} not found in {video_path}")
    shutil.move(video_path, os.path.join(output_dir, 'videos_other'))


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser(description = "SyncNet");
    parser.add_argument('--video_base_dir', '-v', required=True, type=str)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--split_total", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=1.5)

    args = parser.parse_args();

    # ==================== RUN EVALUATION ====================
    syncnet_engine = SyncNetEngine(output_dir=args.video_base_dir, device='cuda');

    video_path = os.path.join(args.video_base_dir, 'videos')
    if not os.path.isdir(video_path):
        syncnet_engine.evaluate(video_path)
    else:
        all_videos = os.listdir(video_path)
        all_videos = [os.path.join(video_path, v) for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        this_split_videos = split_into_parts(all_videos, args.split_total)[args.split_id]
        for vidx, this_video_path in enumerate(this_split_videos):
            print('Processing {}/{}: {}......'.format(vidx+1, len(this_split_videos), this_video_path))
            try:
                syncnet_engine.evaluate(this_video_path)
            except Exception as e:
                print(f'Error processing {this_video_path}: {e}')
                continue
    find_synchronized_videos(args.video_base_dir, args.threshold)
    move_to_new_folder(args.video_base_dir)
