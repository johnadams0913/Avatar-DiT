#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from  Radek Danecek (Max-Planck-Gesellschaft zur Förderung).

import os
import torch 
import numpy as np
from .DecaEncoder import DecaResnetEncoder
from .MicaEncoder import MicaArcfaceEncoder

class EMICAEncoder(torch.nn.Module): 
    def __init__(self, flame_version='2020', device='cpu'):
        super().__init__()
        self._device = device
        self._flame_version = flame_version

    def _init_models(self, ):
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join(_abs_path, f'../../assets/EMICA-CVT_flame{self._flame_version}_notexture.pt')
        assert os.path.exists(_model_path), f"Model not found: {_model_path}."
        state_dict = torch.load(_model_path, map_location='cpu', weights_only=True)
        self.mica_encoder = MicaEncoder()
        self.deca_encoder = DecaEncoder(outsize=86)
        self.expression_encoder = DecaEncoder(outsize=100)
        self.load_state_dict(state_dict, strict=True)
        self.mica_encoder.to(self._device).eval()
        self.deca_encoder.to(self._device).eval()
        self.expression_encoder.to(self._device).eval()

    @torch.inference_mode()
    def forward(self, emica_inputs, with_shape=False):
        if not hasattr(self, 'mica_encoder'):
            self._init_models()
        # emica_inputs = data_to_device(emica_inputs, device=self._device)
        flame_results = self.deca_encoder.encode(emica_inputs['warped_image'].float())
        exp_code = self.expression_encoder.encode(emica_inputs['warped_image'].float())
        if with_shape:
            mica_shape, _ = self.mica_encoder.encode(emica_inputs['mica_image'].float())
            return {
                'shape_params': mica_shape, 'expression_params': exp_code['expcode'], 
                'jaw_params': flame_results['jawpose'], 'pose_params': flame_results['globalpose'], 
            }
        else:
            return {
                'expression_params': exp_code['expcode'], 
                'jaw_params': flame_results['jawpose'], 'pose_params': flame_results['globalpose'], 
            }


class DecaEncoder(torch.nn.Module):
    def __init__(self, outsize=100):
        super().__init__()
        self.encoder = DecaResnetEncoder(outsize=outsize, last_op=None)
        self.encoder.requires_grad_(False)
        if outsize == 100:
            self._prediction_code_dict = {
                'expcode': 100
            }
        elif outsize == 86:
            self._prediction_code_dict = {
                'texcode': 50, 'jawpose': 3, 'globalpose': 3, 'cam': 3, 'lightcode': 27
            }
        else:
            raise ValueError(f"Invalid outsize: {outsize}")

    @torch.no_grad()
    def encode(self, deca_images):
        code_vec = self.encoder(deca_images, output_features=False)
        results = self._decompose_code(code_vec)
        return results
    
    def _decompose_code(self, code):
        '''
        Decompose the code into the different components based on the prediction_code_dict
        '''
        results = {}
        start = 0
        for key, dim in self._prediction_code_dict.items():
            subcode = code[..., start:start + dim]
            if key == 'light':
                subcode = subcode.reshape(subcode.shape[0], 9, 3)
            results[key] = subcode
            start = start + dim
        return results


class MicaEncoder(torch.nn.Module): 
    def __init__(self, ):
        super().__init__()
        self.E_mica = MicaArcfaceEncoder()

    @torch.no_grad()
    def encode(self, mica_image):
        mica_encoding = self.E_mica.encode(mica_image) 
        mica_shapecode, identity_code = self.E_mica.decode(mica_encoding)
        return mica_shapecode, identity_code


def data_to_device(data_dict, device='cuda'):
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.tensor(data_dict[key], device=device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_device(data_dict[key], device=device)
        else:
            continue
    return data_dict
