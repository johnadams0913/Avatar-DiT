import torch
import ipdb
import trimesh
from flame_model import FLAMEModel

face_decoder = FLAMEModel(n_shape=300, n_exp=100)
pred_motion_code = torch.zeros(2, 110) # frames x motion code
pred_verts = face_decoder.get_flame_verts(pred_motion_code[None])[0]
faces = face_decoder.get_faces()
for i in range(pred_verts.shape[0]):
    mesh = trimesh.Trimesh(vertices=pred_verts[i], faces=faces)
    output_filename = f"pred_mesh_{i}.obj"
    mesh.export(output_filename)