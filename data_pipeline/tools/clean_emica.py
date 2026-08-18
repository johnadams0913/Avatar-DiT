import torch

ori_path = '../assets/EMICA-CVT_flame2023_notexture.ckpt'
target_path = '../assets/EMICA-CVT_flame2023_notexture.pt'

state_dict = torch.load(ori_path, map_location='cpu', weights_only=False)['state_dict']
new_state_dict = {}
for key in state_dict.keys():
    if 'face_encoder.mica_deca_encoder.' in key:
        new_state_dict[key.replace('face_encoder.mica_deca_encoder.', '')] = state_dict[key]
    elif 'face_encoder.expression_encoder' in key:
        new_state_dict[key.replace('face_encoder.', '')] = state_dict[key]
    else:
        continue
torch.save(new_state_dict, target_path)
