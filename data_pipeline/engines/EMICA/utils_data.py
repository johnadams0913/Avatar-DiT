#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import numpy as np
import torchvision

from .utils_fan_detector import FANDetector
from .utils_insight_detector import InsightDetector

class EMICAImageEngine:
    def __init__(self, device='cuda'):
        self.device = device
        self.resolution_inp = 224

    def _init_models(self, ):
        self.face_detector = FANDetector(device=self.device)
        self.insight_detector = InsightDetector(device=self.device)

    @torch.inference_mode()
    def __call__(self, image_tensor, image_key='default', with_mica=False):
        if not hasattr(self, 'face_detector'):
            self._init_models()
        # image_tensor: [3, h, w], [0, 255]
        # assert image_tensor.max() >= 1.0, f"Ori image max: {image_tensor.max()}."
        image_tensor = image_tensor.to(self.device)
        if image_tensor.max() < 1.0:
            print(f"Ori image max: {image_tensor.max()}.")
            return None, None, None
        # Warp processing
        warped_image, mica_inp_image, landmarks, bbox = self.warp_processing(image_tensor, image_key, with_mica)

        if warped_image is None:
            return None, None, None
        # assert warped_image.max() <= 1.2, f"Warped image max: {warped_image.max()}."
        if warped_image.max() > 1.2:
            print(f"Warped image max: {warped_image.max()}.")
            return None, None, None
        # MICA processing
        if with_mica:
            mica_image = self.mica_processing(mica_inp_image, image_key)
            if mica_image is None:
                return None, None, None
            # assert mica_image.max() <= 1.2, f"Mica image max: {mica_image.max()}."
            if mica_image.max() > 1.2:
                print(f"Mica image max: {mica_image.max()}.")
                return None, None, None
            emica_inputs = {
                'image':image_tensor.float()/255.0, 'image_key': image_key, 
                'warped_image': warped_image.float(), 'mica_image': mica_image, 
            }
        else:
            emica_inputs = {
                'image':image_tensor.float()/255.0, 'image_key': image_key, 
                'warped_image': warped_image.float()
            }
        gaze_inputs = {'image_ori': image_tensor.cpu().numpy(), 'landmarks': landmarks.cpu().numpy()}
        return emica_inputs, gaze_inputs, bbox

    def warp_processing(self, image_tensor, image_key, with_mica):
        # image_tensor: [3, h, w]
        bbox, landmarks = self.face_detector(image_tensor.to(self.device), with_landmarks=True)

        if bbox is None:
            print('No fan face detected: {}!'.format(image_key))
            return None, None, None, None
        warped_image = crop_and_resize(image_tensor, bbox, t_size=self.resolution_inp, t_scale=1.25)
        if with_mica:
            mica_inp_image = crop_and_resize(image_tensor, bbox, t_size=320, t_scale=2.0)
        else:
            mica_inp_image = None
        return warped_image / 255.0, mica_inp_image, landmarks, bbox

    def mica_processing(self, image_tensor, image_key):
        bbox, kps, score = self.insight_detector.get(image_tensor)
        if bbox is None:
            print('No insight face detected: {}!'.format(image_key))
            return None
        aimg = norm_crop(image_tensor.permute(1, 2, 0).cpu().numpy(), landmark=kps.cpu().numpy())
        aimg = torch.tensor(aimg).permute(2, 0, 1).to(image_tensor.device)
        mica_frame = (aimg - 127.5) / 127.5
        return mica_frame

    @staticmethod
    def load_image(img_path):
        image = torchvision.io.read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB)
        return image


def crop(image_tensor, bbox_xyxy, t_size=512, t_scale=1.25):
    left, top, right, bottom = bbox_xyxy
    target_hw = int((right - left + bottom - top) / 2 * 1.1 * t_scale)
    center_x = (right + left) / 2.0
    center_y = (bottom + top) / 2.0
    target_left = int(center_x - target_hw / 2)
    target_top = int(center_y - target_hw / 2)
    cropped = torchvision.transforms.functional.crop(
        image_tensor, target_top, target_left, target_hw, target_hw
    )
    _, h, w = cropped.shape
    if h == 0 or w == 0:
        return torch.zeros((3, t_size, t_size), dtype=image_tensor.dtype, device=image_tensor.device)
    if h > w:
        new_h, new_w = t_size, int(w * t_size / h)
    else:
        new_h, new_w = int(h * t_size / w), t_size
    cropped = torchvision.transforms.functional.resize(cropped, (new_h, new_w), antialias=True)
    pad_w = t_size - cropped.shape[2]
    pad_h = t_size - cropped.shape[1]
    cropped = torchvision.transforms.functional.pad(
        cropped, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=0
    )
    return cropped


def crop_and_resize(image_tensor, bbox_xyxy, t_size=224, t_scale=1.25):
    left, top, right, bottom = bbox_xyxy
    target_hw = int((right - left + bottom - top) / 2 * 1.1 * t_scale)
    center_x = (right + left) / 2.0
    center_y =  (bottom + top) / 2.0
    target_left = int(center_x - target_hw / 2)
    target_top = int(center_y - target_hw / 2)
    warped_image = torchvision.transforms.functional.crop(
        image_tensor, target_top, target_left, target_hw, target_hw
    )
    warped_image = torchvision.transforms.functional.resize(
        warped_image, size=(t_size, t_size), antialias=True
    )
    return warped_image


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    import cv2
    import skimage
    def estimate_norm(lmk, image_size=112,mode='arcface'):
        arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32
        )
        assert lmk.shape == (5, 2)
        assert image_size%112==0 or image_size%128==0
        if image_size%112==0:
            ratio = float(image_size)/112.0
            diff_x = 0
        else:
            ratio = float(image_size)/128.0
            diff_x = 8.0*ratio
        dst = arcface_dst * ratio
        dst[:,0] += diff_x
        tform = skimage.transform.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        return M
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
