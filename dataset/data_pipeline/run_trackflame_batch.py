#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import av
import os
import json
import ipdb
import torch
import argparse
import torchaudio
import os.path as osp

from engines import FLAMEModel, LMDBEngine
from tqdm import tqdm

def read_video_frames(video_path):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frame_array = frame.to_ndarray(format="rgb24")
        frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1)
        frames.append(frame_tensor)
    return frames

class Tracker:
    def __init__(self, lmdb_path, device='cuda'):
        self._device = device
        self._flame_version = '2020'
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, flame_version=self._flame_version).to(device)
        print(f"lmdb path: {lmdb_path}")
        self.lmdb_engine = LMDBEngine(lmdb_path, write=False)
        print(f'Tracking with FLAME version: {self._flame_version}')

    def run_visualization(self, video_path, shapecode, motioncode, original_video_path=None):
        from engines.utils import read_video_frames, get_video_info, write_video, pad_resize, RenderMesh
        shapecode = torch.tensor(shapecode).to(self._device).float()
        motioncode = torch.tensor(motioncode).to(self._device).float()
        flame_vertices = self.flame_model(
            shape=shapecode.expand(motioncode.shape[0], -1), expression=motioncode[:, :100],
            gpose=motioncode[:, 100:103], jaw_pose=motioncode[:, 103:104], eye_pose=motioncode[:, 106:112],
        )
        mesh_render = RenderMesh(512, faces=self.flame_model.get_faces().cpu().numpy()).to(self._device)

        vis_images = []
        video_length, video_fps = motioncode.shape[0], 30
        if original_video_path is not None:
            video_meta_data = get_video_info(original_video_path)['video']
            video_length, video_fps = video_meta_data["num_frames"], video_meta_data["frame_rate"]
            print(video_meta_data)
            frames = read_video_frames(original_video_path)
        for (frame, verts) in tqdm(zip(frames, flame_vertices), desc=f"video: {osp.basename(video_path)}"):
            tracked_frame, _ = mesh_render(verts[None], colors=self.flame_model.get_colors())
            frame = pad_resize(frame, image_size=512)
            vis_images.append(torch.cat([frame, tracked_frame[0].cpu()], -1))
        # for verts in tqdm(flame_vertices, desc=f"video: {osp.basename(video_path)}"):
        #     tracked_frame, _ = mesh_render(verts[None], colors=self.flame_model.get_colors())
        #     vis_images.append(tracked_frame[0].cpu())
        vis_images = torch.stack(vis_images, dim=0)
        write_video(vis_images, video_path, fps=int(video_fps))

    def close(self):
        if hasattr(self, 'lmdb_engine'):
            self.lmdb_engine.close()


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', default="data_lmdb", type=str)
    parser.add_argument('--output_folder', '-o', required=True, type=str)
    parser.add_argument('--union_dict', default="union_files.json")
    parser.add_argument('--video_path', default=None)
    args = parser.parse_args()

    if not osp.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    track_engine = Tracker(args.input_file)
    union_files = json.load(open(args.union_dict, "r"))
    for video in tqdm(track_engine.lmdb_engine.keys()):
        data = track_engine.lmdb_engine[video]
        if osp.basename(video) in union_files:
            track_engine.run_visualization(
                osp.join(args.output_folder, osp.basename(video) + ".mp4"),
                data["shapecode"],
                data["motioncode"],
                original_video_path = args.video_path
            )