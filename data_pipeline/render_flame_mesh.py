#!/usr/bin/env python

import os
import torch
import argparse
from tqdm.rich import tqdm

from engines import FLAMEModel, LMDBEngine
from engines.utils import write_video, RenderMesh, get_video_info


class FlameRenderer:
    def __init__(self, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, falling back to CPU')
            device = 'cpu'
        self._device = device
        self._flame_version = '2020'
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, flame_version=self._flame_version).to(device)
        print(f'Rendering with FLAME version: {self._flame_version} on device: {device}')

    def render_from_lmdb(self, input_path, output_path, video_folder=None, default_fps=30):
        lmdb_engine = LMDBEngine(input_path, write=False)
        all_keys = lmdb_engine.keys()
        
        if not all_keys:
            print(f'No data found in LMDB: {input_path}')
            lmdb_engine.close()
            return
        
        print(f'Found {len(all_keys)} videos in LMDB')
        os.makedirs(output_path, exist_ok=True)
        
        for key_name in tqdm(all_keys, desc='Rendering videos'):
            try:
                data = lmdb_engine[key_name]
                shapecode = torch.tensor(data['shapecode']).to(self._device).float()
                motioncode = torch.tensor(data['motioncode']).to(self._device).float()
                
                fps = default_fps
                if video_folder is not None:
                    video_path = os.path.join(video_folder, f'{key_name}.mp4')
                    if os.path.exists(video_path):
                        try:
                            video_info = get_video_info(video_path)
                            if video_info["video"] is not None:
                                fps = int(video_info["video"]["frame_rate"])
                        except Exception as e:
                            print(f'Warning: Could not read FPS from {video_path}, using default {default_fps}')
                    else:
                        print(f'Warning: Video file not found {video_path}, using default FPS {default_fps}')
                
                flame_vertices = self.flame_model(
                    shape=shapecode.expand(motioncode.shape[0], -1),
                    expression=motioncode[:, :100],
                    gpose=motioncode[:, 100:103],
                    jaw_pose=motioncode[:, 103:104],
                    eye_pose=motioncode[:, 106:112],
                )
                
                mesh_render = RenderMesh(512, faces=self.flame_model.get_faces().cpu().numpy()).to(self._device)
                
                vis_images = []
                for fidx in range(flame_vertices.shape[0]):
                    rendered_frame, _ = mesh_render(flame_vertices[fidx][None], colors=self.flame_model.get_colors())
                    vis_images.append(rendered_frame[0].cpu())
                
                vis_images = torch.stack(vis_images, dim=0)
                output_file = os.path.join(output_path, f'{key_name}.mp4')
                write_video(vis_images, output_file, fps=fps)
                
            except Exception as e:
                print(f'Error rendering {key_name}: {e}')
                continue
        
        lmdb_engine.close()
        print(f'Rendering completed! Videos saved to: {output_path}')


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', required=True, type=str, help='Path to LMDB file')
    parser.add_argument('--output_path', '-o', required=True, type=str, help='Output directory for rendered videos')
    parser.add_argument('--video_folder', '-v', type=str, default=None, help='Path to original video folder to get FPS')
    parser.add_argument('--fps', type=int, default=30, help='Default FPS if video folder not provided')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    renderer = FlameRenderer(device=args.device)
    renderer.render_from_lmdb(args.input_path, args.output_path, video_folder=args.video_folder, default_fps=args.fps)

