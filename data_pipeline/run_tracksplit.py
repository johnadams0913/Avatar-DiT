#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import argparse

from engines import TrackSplitEngine
from engines.utils import split_into_parts

if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', required=True, type=str)
    parser.add_argument('--output_dir', '-o', required=True, type=str)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--split_total", type=int, default=1)
    args = parser.parse_args()


    track_split_engine = TrackSplitEngine(output_dir=args.output_dir, device='cuda', silent=os.path.isdir(args.video_path))
    if not os.path.isdir(args.video_path):
        track_split_engine.run_video_pipeline(args.video_path)
    else:
        all_videos = os.listdir(args.video_path)
        all_videos = [os.path.join(args.video_path, v) for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        this_split_videos = split_into_parts(all_videos, args.split_total)[args.split_id]
        for vidx, this_video_path in enumerate(this_split_videos):
            print('Processing {}/{}: {}......'.format(vidx+1, len(this_split_videos), this_video_path))
            try:
                track_split_engine.run_video_pipeline(this_video_path)
            except Exception as e:
                print(f'Error processing {this_video_path}: {e}')
                continue
