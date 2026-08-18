#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import random
import pytorch3d
import numpy as np
import torchvision
from tqdm.rich import tqdm
from scipy.signal import savgol_filter
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from .UniGaze import UniGazeEncoder
from .EMICA import EMICAEncoder, EMICAImageEngine
from .EMICA.utils_data import crop
from .utils import read_video_frames, get_video_info
from .engine_trackcrop import expand_bbox

class FLAMEEngine:
    def __init__(self, image_size, focal_length, flame_version='2020', device='cuda'):
        random.seed(42)
        self._device = device
        # paths and data engine
        self.image_size = image_size
        self.focal_length = focal_length
        self.motion_code_dim = 100+3+3+6
        self.gaze_encoder = UniGazeEncoder(device=device)
        self.flame_encoder = EMICAEncoder(flame_version=flame_version, device=device)
        self.emica_data_engine = EMICAImageEngine(device=self._device)

    def track_motion(self, video_path, false_ratio=0.05):
        # get video meta
        video_meta = get_video_info(video_path)['video']
        # frame_size = min(video_meta["width"], video_meta["height"]) # for seamless interaction
        frame_size = min(max(video_meta["width"], video_meta["height"]), 1920)
        print(f'Processing video: {video_path} with frame size: {frame_size}.')
        video_length = video_meta['num_frames']
        # track video
        all_shape, all_motion, all_gaze, bboxes = [], [], [], []

        counter, non_limitation = 0, 900
        frames = []
        for fidx, frame in enumerate(tqdm(read_video_frames(video_path), total=video_length)):
            frame = frame[:, :frame_size, :frame_size]  # for seamless interaction
            with_shape = (fidx % (video_length//12) == 0 and len(all_shape) < 10)
            frame = pad_resize(frame, image_size=frame_size)
            frames.append(frame)
            # flame expression
            emica_inputs, gaze_inputs, bbox = self.emica_data_engine(frame, with_mica=with_shape)
            if emica_inputs is None:
                flame_outputs = None
            else:
                emica_inputs = torch.utils.data.default_collate([emica_inputs])
                flame_outputs = self.flame_encoder(emica_inputs, with_shape=with_shape)
                # head rotation
                gaze_inputs['pose_params'] = flame_outputs['pose_params']
                gaze_outputs = self.gaze_encoder(gaze_inputs)
            if flame_outputs is None:
                all_motion.append(torch.ones(1, self.motion_code_dim, device=self._device) * float('nan'))
                counter += 1
                if counter > non_limitation:
                    return None, None
                continue
            # gather results
            if with_shape:
                all_shape.append(flame_outputs['shape_params'])
            all_motion.append(
                torch.cat([flame_outputs['expression_params'], flame_outputs['pose_params'], flame_outputs['jaw_params'], gaze_outputs], dim=-1)
            )
            if bbox is not None:
                bboxes.append(bbox)

        if len(bboxes) == 0:
            return None, None
        
        valid_bboxes = [bbox for bbox in bboxes if bbox is not None]
        if len(valid_bboxes) == 0:
            return None, None
            
        bboxes_array = np.array(valid_bboxes)
        x1_mean = np.mean(bboxes_array[:, 0])
        y1_mean = np.mean(bboxes_array[:, 1]) 
        x2_mean = np.mean(bboxes_array[:, 2])
        y2_mean = np.mean(bboxes_array[:, 3])
        
        center_box = [x1_mean, y1_mean, x2_mean, y2_mean]
        center_box = expand_bbox(center_box, video_meta['height'], video_meta['width'], bbox_scale=1.1)
        all_cropped = []
        for frame in frames:
            all_cropped.append(crop(frame, center_box, t_size=512))

        if len(all_shape) == 0 or len(all_motion) == 0:
            return None, None
        shapecode = torch.cat(all_shape, dim=0).mean(dim=0, keepdim=True).cpu()
        motioncode = torch.cat(all_motion, dim=0).cpu()
        cropped = torch.stack(all_cropped, dim=0).cpu()

        if torch.isnan(motioncode).any():
            print('Interpolating missing motion for {}.'.format(video_path))
            motioncode = interpolate_missing_motion(motioncode)
        assert not torch.isnan(motioncode).any(), 'Motion code contains NaN values.'
        return shapecode, motioncode, cropped

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        motion_np = motion_codes.clone().detach().cpu().numpy()
        motion_np = savgol_filter(motion_np, window_length=5, polyorder=3, axis=0)
        motion_smoothed = torch.tensor(motion_np).type_as(motion_codes)
        return motion_smoothed

    @staticmethod
    def smooth_globalpose_savgol(motion_codes):
        motion_np = motion_codes.clone().detach().cpu().numpy()
        motion_np[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=9, polyorder=3, axis=0)
        motion_np[..., 106:] = savgol_filter(motion_np[..., 106:], window_length=11, polyorder=3, axis=0)
        motion_smoothed = torch.tensor(motion_np).type_as(motion_codes)
        return motion_smoothed


def interpolate_missing_motion(motioncode):
    np_data = motioncode.cpu().clone().numpy()
    x = np.arange(np_data.shape[0])
    mask = np.isnan(np_data).all(axis=1)

    for i in range(np_data.shape[1]):
        y = np_data[:, i]
        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) > 1:
            np_data[:, i] = np.interp(x, valid_idx, y[valid_idx])

    return torch.tensor(np_data).type_as(motioncode)


def pad_resize(image, image_size=512):
    _, h, w = image.shape
    if h > w:
        new_h, new_w = image_size, int(w * image_size / h)
    else:
        new_h, new_w = int(h * image_size / w), image_size
    image = torchvision.transforms.functional.resize(image, (new_h, new_w), antialias=True)
    pad_w = image_size - image.shape[2]
    pad_h = image_size - image.shape[1]
    image = torchvision.transforms.functional.pad(image, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=0)
    return image
