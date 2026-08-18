#!/usr/bin/env python
# Copyright (c) Ruicong Liu and Xuangeng Chu (xg.chu@outlook.com)
import os
import cv2
import torch
import numpy as np
import torchvision

from .models import MAE_Gaze
from .face_model import face_model
from .gaze_utils import normalize, estimateHeadPose, get_face_center_by_nose, set_dummy_camera_model, denormalize_predicted_gaze

class UniGazeEncoder(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self._device = device
        # model
        self.model = MAE_Gaze(model_type='vit_b_16', global_pool=False, drop_path_rate=0.1)
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join(_abs_path, '../../assets/unigaze_b16_joint.pth.tar')
        assert os.path.exists(_model_path), f"Model not found: {_model_path}."
        state_dict = torch.load(_model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict['model_state'], strict=True)
        self.model.to(self._device).eval()
        # face model
        self.face_model = face_model
        # normalization
        self.focal_norm = 960 # focal length of normalized camera
        self.distance_norm = 600  # normalized distance between eye and camera
        self.roi_size = (224, 224)  # size of cropped eye image
        self.image_torch_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def forward(self, gaze_inputs):
        image_origin = gaze_inputs['image_ori'].copy()
        landmarks = gaze_inputs['landmarks'].copy()
        pose_params = gaze_inputs['pose_params'].cpu().numpy().copy()
        pose_params[:, 1] *= -1

        x_min = int(landmarks[:, 0].min())
        x_max = int(landmarks[:, 0].max())
        y_min = int(landmarks[:, 1].min())
        y_max = int(landmarks[:, 1].max())

        scale_factor = 2.0
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        image_height, image_width = image_origin.shape[1], image_origin.shape[2]
        bbox_center = ( (x_min + x_max) // 2, (y_min + y_max) // 2 )
        x_min = max(0, bbox_center[0] - int(bbox_width * scale_factor // 2))
        x_max = min(image_width, bbox_center[0] + int(bbox_width * scale_factor // 2))
        y_min = max(0, bbox_center[1] - int(bbox_height * scale_factor // 2))
        y_max = min(image_height, bbox_center[1] + int(bbox_height * scale_factor // 2))

        image = image_origin[:, y_min:y_max, x_min:x_max].transpose(1, 2, 0)
        landmarks = landmarks - np.array([x_min, y_min])

        ################# normalization #################
        camera_matrix, camera_distortion = set_dummy_camera_model(image=image)
        face_model = self.face_model[[20, 23, 26, 29, 15, 19], :]
        facePts = face_model.reshape(6, 1, 3)

        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub_paint = landmarks_sub
        landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
        hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

        # hR0 = cv2.Rodrigues(hr)[0]
        hR = cv2.Rodrigues(pose_params[0])[0]  # rotation matrix
        face_center_camera_cord, Fc_nose = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=self.face_model)

        # -------------------------------------------- normalize image --------------------------------------------
        img_normalized, R, hR_norm, gaze_normalized, landmarks_normalized, _ = normalize(image, landmarks, self.focal_norm, self.distance_norm, self.roi_size, face_center_camera_cord, hr, ht, camera_matrix, gc=None)
        hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
        # if np.linalg.norm(hr_norm) > 80 * np.pi / 180 :
        #     continue
        input_var = self.image_torch_transform(img_normalized)
        # input_var = torch.autograd.Variable(input_var.float().to(self._device))
        input_var = input_var.to(self._device).unsqueeze(0).float()
        # torchvision.utils.save_image(input_var, 'input_var.png')
        ret = self.model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        
        pred_gaze = ret["pred_gaze"][0]
        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array

        R_inv = np.linalg.inv(R)
        pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor = denormalize_predicted_gaze(pred_gaze_np, R_inv)
        
        # 计算头部坐标系下的gaze
        pred_gaze_head_coord = np.dot(hR.T, pred_gaze_cancel_nor)  # 使用头部旋转矩阵的转置进行坐标变换
        pred_gaze_head_coord = pred_gaze_head_coord / np.linalg.norm(pred_gaze_head_coord)  # 归一化
        x, y, z = pred_gaze_head_coord
        pred_gaze_head_coord[0] = -x
        pred_gaze_head_coord[1] = y
        pred_gaze_head_coord[2] = z

        gaze_axis_angle = gaze_vector_to_flame_axis_angle(pred_gaze_head_coord)
        gaze_axis_angle = torch.tensor(np.concatenate([gaze_axis_angle, gaze_axis_angle], axis=0), device=self._device)[None]
        return gaze_axis_angle


def gaze_vector_to_flame_axis_angle(gaze_head_coord, reference_direction=None):
    """
    将头部坐标系下的gaze向量转换为FLAME模型的轴角表示
    
    Args:
        gaze_head_coord: 头部坐标系下的gaze向量 (3,1) 或 (3,)
        reference_direction: 参考方向，通常是头部的前向方向 [0, 0, 1]，如果为None则使用默认值
        
    Returns:
        axis_angle: FLAME格式的轴角表示 (3,)，表示从参考方向到gaze方向的旋转
    """
    # 确保输入是列向量
    if gaze_head_coord.ndim == 1:
        gaze_head_coord = gaze_head_coord.reshape(3, 1)
    
    # 设置参考方向（头部坐标系下的前向方向）
    if reference_direction is None:
        reference_direction = np.array([0, 0, 1]).reshape(3, 1)
    else:
        if reference_direction.ndim == 1:
            reference_direction = reference_direction.reshape(3, 1)
    
    # 归一化参考方向
    reference_direction = reference_direction / np.linalg.norm(reference_direction)
    
    # 计算从参考方向到gaze方向的旋转
    # 使用叉积计算旋转轴
    rotation_axis = np.cross(reference_direction.flatten(), gaze_head_coord.flatten())
    axis_norm = np.linalg.norm(rotation_axis)
    
    # 如果旋转轴接近零向量，说明gaze方向与参考方向几乎相同
    if axis_norm < 1e-6:
        # 返回零旋转
        return np.zeros(3)
    
    # 归一化旋转轴
    rotation_axis = rotation_axis / axis_norm
    
    # 计算旋转角度（使用点积）
    cos_angle = np.dot(reference_direction.flatten(), gaze_head_coord.flatten())
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
    angle = np.arccos(cos_angle)
    
    # 构建轴角表示：axis * angle
    axis_angle = rotation_axis * angle
    
    return axis_angle
