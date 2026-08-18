"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Modified from smplx code for FLAME by Xuangeng Chu (xg.chu@outlook.com)
"""
import os
import torch
import torch.nn as nn

from .lbs import lbs

class FLAMEModel(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, n_shape, n_exp, flame_version='2020'):
        super().__init__()
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        self.flame_path = os.path.join(_abs_path, '../../assets')
        self.flame_ckpt = torch.load(os.path.join(self.flame_path, f'flame{flame_version}.pt'), weights_only=True)
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', self.flame_ckpt['f'])
        self.register_buffer('v_template', self.flame_ckpt['v_template'])
        shapedirs = self.flame_ckpt['shapedirs']
        self.register_buffer('shapedirs', torch.cat([shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_exp]], 2))
        num_pose_basis = self.flame_ckpt['posedirs'].shape[-1]
        self.register_buffer('posedirs', self.flame_ckpt['posedirs'].reshape(-1, num_pose_basis).T)
        self.register_buffer('J_regressor', self.flame_ckpt['J_regressor'])
        parents = self.flame_ckpt['kintree_table'][0]
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', self.flame_ckpt['weights'])
        # Fixing Eyeball and neck rotation
        self.register_buffer('eye_pose', torch.zeros([1, 6], dtype=torch.float32))
        self.register_buffer('neck_pose', torch.zeros([1, 3], dtype=torch.float32))
        # eye_vertices = list(range(4477, 4594, 4)) + [4598, 4602, 4597] + \
        #                list(range(3931, 4048, 4)) + [4052, 4056, 4051]
        eye_vertices = list(range(4477, 4594, 1)) + [4598, 4602, 4597] + \
                       list(range(3931, 4048, 1)) + [4052, 4056, 4051]
        verts_rgb = torch.ones_like(self.v_template) * torch.tensor([142, 179, 247])[None, None]  # (1, V, 3)
        # verts_rgb[:, eye_vertices, :] = torch.tensor([21, 60, 122])[None, None].float()
        verts_rgb[:, eye_vertices, :] = torch.tensor([255, 0, 0])[None, None].float()
        self.register_buffer('verts_rgb', verts_rgb[0])

    def get_faces(self, ):
        return self.faces_tensor.long()

    def get_colors(self, ):
        return self.verts_rgb

    def forward(self, shape=None, expression=None, gpose=None, jaw_pose=None, eye_pose=None):
        """
            Input:
                shape: N X number of shape parameters
                expression: N X number of expression parameters
                gpose: N X number of global pose parameters (3)
                jaw_pose: N X number of j parameters (3)
                eye_pose: N X number of eye pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0] if shape is not None else expression.shape[0]
        if shape is None:
            shape = self.v_template.new_zeros(batch_size, self.n_shape)
        if expression is None:
            expression = self.v_template.new_zeros(batch_size, self.n_exp)
        if gpose is None:
            gpose = self.v_template.new_zeros(batch_size, 3)
        if jaw_pose is None:
            jaw_pose = self.v_template.new_zeros(batch_size, 3)
        if eye_pose is None:
            eye_pose = self.v_template.new_zeros(batch_size, 6)
        if jaw_pose.shape[1] == 1:
            jaw_pose = torch.cat([jaw_pose, jaw_pose.new_zeros(batch_size, 2)], dim=1)

        # build flame
        betas = torch.cat([shape, expression], dim=1)
        full_pose = torch.cat([gpose, self.neck_pose.expand(batch_size, -1), jaw_pose, eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        vertices, head_joints = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs, self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype, detach_pose_correctives=False
        )
        return vertices
