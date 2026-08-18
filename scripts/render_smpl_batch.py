"""
Batch SMPL render for multiple DNA cases.
Run inside mvperformer env.

Each case <case_id>:
  ./data/gt_data/dna/<case_id>/
    cam.pkl
    smpl_mesh/*.obj (49 obj)
    crop_gt/cam_XX/  (per-cam GT png sequence)

Output:
  ./output/dna_smpl_renders/<case_id>/smpl_cam_XX.mp4
"""
import os
import os.path as osp
import pickle
import argparse

import cv2
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, MeshRenderer, MeshRasterizer, SoftPhongShader,
    RasterizationSettings, PointLights, TexturesVertex, BlendParams,
)


CASES_DNA = [
    "0012_09", "0019_06", "0025_11", "0034_04", "0094_02",
    "0124_03", "0152_01", "0165_08", "0188_02", "0219_07",
]
TARGET_W, TARGET_H = 512, 768
FPS = 24
MESH_COLOR = (0.80, 0.82, 0.90)


def load_meshes(mesh_dir, device="cuda"):
    files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".obj")])
    verts_list, faces_list = [], []
    for fn in files:
        v, f, _ = load_obj(osp.join(mesh_dir, fn), load_textures=False)
        verts_list.append(v.to(device))
        faces_list.append(f.verts_idx.to(device))
    return files, verts_list, faces_list


def resize_with_crop_np(img_np, target_w, target_h):
    h, w = img_np.shape[:2]
    img_ratio = h / w
    target_ratio = target_h / target_w
    if img_ratio > target_ratio:
        scale = target_w / w
        new_h = int(round(h * scale))
        scaled = cv2.resize(img_np, (target_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        start_y = max(0, new_h // 2 - target_h // 2)
        cropped = scaled[start_y:start_y + target_h]
    else:
        scale = target_h / h
        new_w = int(round(w * scale))
        scaled = cv2.resize(img_np, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        start_x = max(0, new_w // 2 - target_w // 2)
        cropped = scaled[:, start_x:start_x + target_w]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def render_for_cam(verts_list, faces_list, K, w2c, render_H, render_W, device="cuda"):
    R_cv = torch.tensor(w2c[:3, :3], dtype=torch.float32, device=device)
    t_cv = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=device)
    R_z180 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32, device=device)
    R_p3d = (R_cv.T @ R_z180).unsqueeze(0)
    T_p3d = (R_z180 @ t_cv).unsqueeze(0)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cameras = PerspectiveCameras(
        R=R_p3d, T=T_p3d,
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor([[render_H, render_W]], dtype=torch.float32, device=device),
        in_ndc=False, device=device,
    )
    rs = RasterizationSettings(image_size=(render_H, render_W), blur_radius=0.0,
                                faces_per_pixel=1, bin_size=0, max_faces_per_bin=300000)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=rs)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]],
                          ambient_color=((0.85, 0.85, 0.85),),
                          diffuse_color=((0.25, 0.25, 0.25),),
                          specular_color=((0.0, 0.0, 0.0),))
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights,
                              blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    out_frames = []
    for v, f in zip(verts_list, faces_list):
        verts_rgb = torch.tensor(MESH_COLOR, device=device).expand_as(v)[None]
        mesh = Meshes(verts=[v], faces=[f], textures=TexturesVertex(verts_features=verts_rgb))
        img = renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        img_resized = resize_with_crop_np(img_uint8, TARGET_W, TARGET_H)
        out_frames.append(img_resized)
    return out_frames


def render_case(case_id, gt_root, out_dir, device="cuda"):
    case_dir = osp.join(gt_root, case_id)
    mesh_dir = osp.join(case_dir, "smpl_mesh")
    cam_pkl = osp.join(case_dir, "cam.pkl")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[{case_id}] loading {len(os.listdir(mesh_dir))} meshes...")
    _, verts_list, faces_list = load_meshes(mesh_dir, device=device)
    print(f"[{case_id}] loaded {len(verts_list)} meshes")

    with open(cam_pkl, "rb") as f:
        cams = pickle.load(f)

    target_cams = [f"cam_{i:02d}" for i in range(16)]
    for cid in target_cams:
        out_path = osp.join(out_dir, f"smpl_{cid}.mp4")
        if osp.exists(out_path):
            print(f"  skip existing {out_path}")
            continue
        c = cams[cid]
        K = np.asarray(c["K"], dtype=np.float64)
        w2c = np.asarray(c["w2c"], dtype=np.float64)
        render_W = int(round(K[0, 2] * 2))
        render_H = int(round(K[1, 2] * 2))
        frames = render_for_cam(verts_list, faces_list, K, w2c, render_H, render_W, device=device)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                  (TARGET_W, TARGET_H))
        for f_rgb in frames:
            writer.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  saved {out_path} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=None,
                   help="cases to render (default: all 10 dna)")
    p.add_argument("--gt_root", default="./data/gt_data/dna")
    p.add_argument("--out_root", default="./output/dna_smpl_renders")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = args.cases or CASES_DNA
    for case_id in cases:
        out_dir = osp.join(args.out_root, case_id)
        render_case(case_id, args.gt_root, out_dir, device=device)


if __name__ == "__main__":
    main()
