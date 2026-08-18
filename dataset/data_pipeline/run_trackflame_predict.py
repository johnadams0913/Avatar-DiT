#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import json
import torch
import argparse
# import torchaudio
import os.path as osp

from tqdm.rich import tqdm
from glob import glob

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

    def track_motion(self, video_path, save_path):
        # check video
        try:
            video_info = get_video_info(video_path)
        except Exception as e:
            print('Error occurred when reading video: {}!'.format(video_path))
            print(e)
            return False
        if video_info["video"] is None:
            print('No video or audio detected in the video: {}!'.format(video_path))
            return False

        # check if video is already tracked
        video_key_name = os.path.basename(video_path).replace('.mp4', '')
        if hasattr(self, 'lmdb_engine') and self.lmdb_engine.exists(video_key_name):
            print('Video {} already tracked!'.format(video_path))
            return True

        # get audio
        # audio_data, audio_rate = read_audio_samples(video_path)
        # if audio_data is None:
            # print('No audio detected in the video: {}!'.format(video_path))
            # return False
        # audio_data = torch.tensor(audio_data).float()
        # audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=16000)
        # audio_data = audio_data.numpy().astype('float16')

        # track video
        shapecode, motioncode, _cropped = self.tracker.track_motion(video_path)
        if shapecode is None:
            print('No face detected in the video: {}!'.format(video_path))
            return False

        # smooth motion
        motioncode = self.tracker.smooth_motion_savgol(motioncode)
        motioncode = self.tracker.smooth_globalpose_savgol(motioncode)

        # face rotation
        motioncode[:, 100:103] = motioncode[:,
                                 100:103] - motioncode[:, 100:103].mean(dim=0, keepdim=True) #head pose mean point

        motioncode[:, 103:104] = torch.norm(motioncode[:, 103:106] * torch.tensor([1, 0.8, 0.8]).type_as(motioncode),
                                            dim=1, keepdim=True)
        motioncode[:, 104:106] = torch.zeros_like(motioncode[:, 104:106])

        motioncode[:, 106:112] = motioncode[:, 106:112] - motioncode[:, 106:112].mean(dim=0, keepdim=True)
        motioncode[:, 106:112] = batch_naturalize_eyemotion_code(motioncode[:, 106:112])

        shapecode = shapecode.numpy().astype('float16')
        motioncode = motioncode.numpy().astype('float16')

        # save to lmdb
        if hasattr(self, 'lmdb_engine'):
            self.lmdb_engine.dump(video_key_name,
                                  {'shapecode': shapecode, 'motioncode': motioncode})#, 'audio': audio_data})

        if self._debug:
            self.run_visualization(video_path, save_path, shapecode, motioncode)

        return True

    def run_visualization(self, video_path, save_path, shapecode, motioncode):
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
        bn = osp.basename(video_path)
        # write_video(cropped_frames, f"{osp.join(save_path, 'face_cropped', bn)}", fps=int(video_fps))
        write_video(vis_images, f"{osp.join(save_path, 'face_smpl', bn)}", fps=int(video_fps))

    def close(self):
        if hasattr(self, 'lmdb_engine'):
            self.lmdb_engine.close()


def predict_face_motion(video_path, lmdb_path, save_path, debug):
    track_flame_engine = Tracker(lmdb_path, device='cuda', debug=debug)
    if not os.path.isdir(video_path) and not video_path.endswith('.json'):
        track_flame_engine.track_motion(video_path, save_path)   
    elif args.video_path.endswith('.json'):
        pairs = json.load(open(video_path, 'r'))
        root_dir = os.path.dirname(video_path)
        
        pairs_list = list(pairs.items())
        pairs_list.sort(key=lambda x: x[0])
        
        for session, vid_pair in this_split_pairs:
            ret1 = track_flame_engine.track_motion(osp.join(root_dir, f'{vid_pair[0]}.mp4'), save_path)
            if not ret1:
                continue
            ret2 = track_flame_engine.track_motion(osp.join(root_dir, f'{vid_pair[1]}.mp4'), save_path)
            if not ret2:
                track_flame_engine.lmdb_engine.delete(vid_pair[0])
    else:
        all_videos = os.listdir(video_path)
        all_videos = [osp.join(video_path, v) for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        this_split_videos = split_into_parts(all_videos, args.split_total)[args.split_id]
        for vidx, video_path in enumerate(this_split_videos):
            print('Processing {}/{}: {}......'.format(vidx+1, len(this_split_videos), video_path))
            try:
                track_flame_engine.track_motion(video_path, save_path)
            except Exception as e:
                print('Error occurred when tracking video: {}!'.format(video_path))
                print(e)
                continue
    track_flame_engine.close()


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True, type=str)
    parser.add_argument('--output_dir', '-o', required=True, type=str)
    parser.add_argument('--lmdb_path', '-l', required=True, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_num", "-bn", type=int, default=1)
    parser.add_argument("--batch_id", "-bi", type=int, default=0)
    args = parser.parse_args()

    video_files = sorted(glob(osp.join(args.input_dir, '*.mp4'), recursive=True))
    video_num = len(video_files)
    batch_size = video_num // args.batch_num
    video_files = video_files[batch_size * args.batch_id: min(batch_size * (args.batch_id + 1), video_num)]

    face_dir = osp.join(args.output_dir, "face_cropped")
    smpl_dir = osp.join(args.output_dir, "face_smpl")
    if not osp.exists(face_dir):
        os.makedirs(face_dir, exist_ok=True)
    if not osp.exists(smpl_dir):
        os.makedirs(smpl_dir, exist_ok=True)

    for file in tqdm(video_files):
        predict_face_motion(file, args.lmdb_path, args.output_dir, args.debug)