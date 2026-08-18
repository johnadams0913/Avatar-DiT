#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import ipdb
import json
import torch
import argparse
import torchaudio
from tqdm.rich import tqdm

from engines import FLAMEModel, FLAMEEngine, LMDBEngine
from engines.utils import split_into_parts, read_audio_samples, get_video_info
from engines import batch_naturalize_eyemotion_code

class Tracker:
    def __init__(self, lmdb_path, device='cuda', debug=False):
        self._debug = debug
        self._device = device
        self._flame_version = '2020'
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, flame_version=self._flame_version).to(device)
        self.tracker = FLAMEEngine(image_size=512, focal_length=12.0, flame_version=self._flame_version, device=device)
        if not self._debug:
            self.lmdb_engine = LMDBEngine(lmdb_path, write=True)
        print(f'Tracking with FLAME version: {self._flame_version}')

    def track_motion(self, video_path):
        # check video
        try:
            video_info = get_video_info(video_path)
        except Exception as e:
            print('Error occurred when reading video: {}!'.format(video_path))
            print(e)
            return False
        if video_info["video"] is None or video_info["audio"] is None:
            print('No video or audio detected in the video: {}!'.format(video_path))
            return False
        
        # check if video is already tracked
        video_key_name = os.path.basename(video_path).replace('.mp4', '')
        if hasattr(self, 'lmdb_engine') and self.lmdb_engine.exists(video_key_name):
            print('Video {} already tracked!'.format(video_path))
            return True
        
        # get audio
        audio_data, audio_rate = read_audio_samples(video_path)
        if audio_data is None:
            print('No audio detected in the video: {}!'.format(video_path))
            return False
        audio_data = torch.tensor(audio_data).float()
        audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=16000)
        audio_data = audio_data.numpy().astype('float16')
        
        # track video
        shapecode, motioncode, cropped_frames = self.tracker.track_motion(video_path)
        if shapecode is None:
            print('No face detected in the video: {}!'.format(video_path))
            return False
        
        # smooth motion
        motioncode = self.tracker.smooth_motion_savgol(motioncode)
        motioncode = self.tracker.smooth_globalpose_savgol(motioncode)

        # face rotation
        motioncode[:, 100:103] = motioncode[:, 100:103] # - motioncode[:, 100:103].mean(dim=0, keepdim=True) head pose mean point

        motioncode[:, 103:104] = torch.norm(motioncode[:, 103:106] * torch.tensor([1, 0.8, 0.8]).type_as(motioncode), dim=1, keepdim=True)
        motioncode[:, 104:106] = torch.zeros_like(motioncode[:, 104:106])

        motioncode[:, 106:112] = motioncode[:, 106:112] - motioncode[:, 106:112].mean(dim=0, keepdim=True)
        motioncode[:, 106:112] = batch_naturalize_eyemotion_code(motioncode[:, 106:112])

        shapecode = shapecode.numpy().astype('float16')
        motioncode = motioncode.numpy().astype('float16')

        # save to lmdb
        if hasattr(self, 'lmdb_engine'):
            self.lmdb_engine.dump(video_key_name, {'shapecode': shapecode, 'motioncode': motioncode, 'audio': audio_data})

        if self._debug:
            self.run_visualization(video_path, shapecode, motioncode, cropped_frames)
        
        return True

    def run_visualization(self, video_path, shapecode, motioncode, cropped_frames):
        from engines.utils import read_video_frames, get_video_info, write_video, pad_resize, RenderMesh
        shapecode = torch.tensor(shapecode).to(self._device).float()
        motioncode = torch.tensor(motioncode).to(self._device).float()
        flame_vertices = self.flame_model(
            shape=shapecode.expand(motioncode.shape[0], -1), expression=motioncode[:, :100], 
            gpose=motioncode[:, 100:103], jaw_pose=motioncode[:, 103:104], eye_pose=motioncode[:, 106:112],
        )
        mesh_render = RenderMesh(512, faces=self.flame_model.get_faces().cpu().numpy()).to(self._device)
        
        vis_images = []
        video_meta_data = get_video_info(video_path)['video']
        video_length, video_fps = video_meta_data['num_frames'], video_meta_data['frame_rate']
        for fidx, frame in enumerate(tqdm(read_video_frames(video_path), total=video_length)):
            # frame = frame.to(device=self._device).float()
            # frame = pad_resize(frame, image_size=512)
            tracked_frame, _ = mesh_render(flame_vertices[fidx][None], colors=self.flame_model.get_colors())
            # vis_images.append(torch.cat([frame, tracked_frame[0]], dim=-1).cpu())
            vis_images.append(tracked_frame[0].cpu())

        vis_images = torch.stack(vis_images, dim=0)
        write_video(vis_images, "debug.mp4", fps=int(video_fps))
        write_video(cropped_frames, "debug_crop.mp4", fps=int(video_fps))

    def close(self):
        if hasattr(self, 'lmdb_engine'):
            self.lmdb_engine.close()


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', required=True, type=str)
    parser.add_argument('--output_dir', '-o', required=True, type=str)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--split_total", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    track_flame_engine = Tracker(args.output_dir, device='cuda', debug=args.debug)
    if not os.path.isdir(args.video_path) and not args.video_path.endswith('.json'):
        track_flame_engine.track_motion(args.video_path)   
    elif args.video_path.endswith('.json'):
        pairs = json.load(open(args.video_path, 'r'))
        root_dir = os.path.dirname(args.video_path)
        
        pairs_list = list(pairs.items())
        pairs_list.sort(key=lambda x: x[0])
        this_split_pairs = split_into_parts(pairs_list, args.split_total)[args.split_id]
        
        for session, vid_pair in this_split_pairs:
            ret1 = track_flame_engine.track_motion(os.path.join(root_dir, f'{vid_pair[0]}.mp4'))
            if not ret1:
                continue
            ret2 = track_flame_engine.track_motion(os.path.join(root_dir, f'{vid_pair[1]}.mp4'))
            if not ret2:
                track_flame_engine.lmdb_engine.delete(vid_pair[0])
    else:
        all_videos = os.listdir(args.video_path)
        all_videos = [os.path.join(args.video_path, v) for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        this_split_videos = split_into_parts(all_videos, args.split_total)[args.split_id]
        for vidx, video_path in enumerate(this_split_videos):
            print('Processing {}/{}: {}......'.format(vidx+1, len(this_split_videos), video_path))
            try:
                track_flame_engine.track_motion(video_path)
            except Exception as e:
                print('Error occurred when tracking video: {}!'.format(video_path))
                print(e)
                continue
    track_flame_engine.close()
