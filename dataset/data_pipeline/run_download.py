#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import argparse

from engines import DownloadEngine
from engines.utils import split_into_parts

if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    from tqdm.rich import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_ids_path', '-v', required=True, type=str)
    parser.add_argument('--output_dir', '-o', required=True, type=str)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--split_total", type=int, default=1)
    args = parser.parse_args()
    
    # read video ids
    with open(args.video_ids_path, 'r') as f:
        youtube_video_ids = f.readlines()
    youtube_video_ids = [video_id.strip() for video_id in youtube_video_ids]
    youtube_video_ids = sorted(youtube_video_ids)
    print(f'Total {len(youtube_video_ids)} video ids read.')
    this_split_videos = split_into_parts(youtube_video_ids, args.split_total)[args.split_id]

    download_engine = DownloadEngine(output_dir=args.output_dir)
    print(f'Downloading {len(this_split_videos)} videos...')
    for video_id in tqdm(this_split_videos):
        download_engine.run_download(video_id)
    print(f'Merging {len(this_split_videos)} videos...')
    for video_id in tqdm(this_split_videos):
        download_engine.run_merge_raw_video(video_id, remove_raw=True)
