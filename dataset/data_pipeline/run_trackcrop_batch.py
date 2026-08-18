#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import argparse
from glob import glob
import os.path as osp

from engines.engine_trackcrop import TrackCropEngine
from engines.utils import split_into_parts

if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True, type=str)
    parser.add_argument('--output_dir', '-o', required=True, type=str)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--split_total", type=int, default=1)
    args = parser.parse_args()

    video_files = sorted(glob(osp.join(args.input_dir, '*.mp4'), recursive=True))
    for video_file in video_files:
        track_split_engine = TrackCropEngine(output_dir=args.output_dir, device='cuda', silent=os.path.isdir(video_file))
        track_split_engine.run_video_pipeline(video_file)
    # if not os.path.isdir(video_file):
        # track_split_engine.run_video_pipeline(video_file)
    # else:
        # all_videos = os.listdir(video_file)
        # all_videos = [os.path.join(video_file, v) for v in all_videos if v.endswith('.mp4')]
        # all_videos = sorted(all_videos)
        # this_split_videos = split_into_parts(all_videos, args.split_total)[args.split_id]
        # for vidx, this_video_path in enumerate(this_split_videos):
            # print('Processing {}/{}: {}......'.format(vidx+1, len(this_split_videos), this_video_path))
            # try:
                # track_split_engine.run_video_pipeline(this_video_path)
            # except Exception as e:
                # print(f'Error processing {this_video_path}: {e}')
                # continue