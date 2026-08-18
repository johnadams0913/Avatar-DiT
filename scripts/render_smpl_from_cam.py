"""
Render SMPL mesh sequence under the real DNA camera calibrations.
For each target cam_id, produce one SMPL pose mp4 (white-grey body, white bg)
at the model training resolution (512x768).

Run inside `mvperformer` env (has pytorch3d 0.7.8):
    python scripts/render_smpl_from_cam.py
"""
import os
import os.path as osp
import pickle
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, MeshRenderer, MeshRasterizer, SoftPhongShader,
    RasterizationSettings, PointLights, TexturesVertex, BlendParams,
)


def load_meshes(mesh_dir, device="cuda"):
    files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".obj")])
    verts_list, faces_list = [], []
    for fn in files:
        v, f, _ = load_obj(osp.join(mesh_dir, fn), load_textures=False)
        verts_list.append(v.to(device))
        faces_list.append(f.verts_idx.to(device))
    return files, verts_list, faces_list


def resize_with_crop_np(img_np, target_w, target_h):
    """Center crop + resize like backend.functool.resize."""
    h, w = img_np.shape[:2]
    img_ratio = h / w
    target_ratio = target_h / target_w
    if img_ratio > target_ratio:
        # tall image -> scale to match width, crop height
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


def render_for_cam(verts_list, faces_list, K, w2c, render_H, render_W,
                   target_W=512, target_H=768, device="cuda"):
    """K, w2c are np arrays. Render at original (render_H, render_W) then crop+resize to (target_H, target_W)."""
    # Follow MV-Performer convention (scripts/extract_dna_partial_render.py:188-200):
    # R_p3d = w2c[:3,:3].T @ R_z180; T_p3d = R_z180 @ w2c[:3,3]
    # where R_z180 = rotation matrix around Z by 180° (i.e. diag(-1, -1, 1))
    R_cv = torch.tensor(w2c[:3, :3], dtype=torch.float32, device=device)
    t_cv = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=device)
    R_z180 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32, device=device)
    R_p3d = (R_cv.T @ R_z180).unsqueeze(0)  # (1, 3, 3)
    T_p3d = (R_z180 @ t_cv).unsqueeze(0)    # (1, 3)

    # K is in pixel coords; pytorch3d PerspectiveCameras needs focal_length and principal_point in pixel
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cameras = PerspectiveCameras(
        R=R_p3d, T=T_p3d,
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor([[render_H, render_W]], dtype=torch.float32, device=device),
        in_ndc=False,
        device=device,
    )

    rs = RasterizationSettings(image_size=(render_H, render_W), blur_radius=0.0,
                                faces_per_pixel=1, bin_size=0, max_faces_per_bin=300000)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=rs)
    # Ambient light to match training data SMPL look (mostly flat)
    lights = PointLights(device=device,
                          location=[[0.0, 0.0, -3.0]],
                          ambient_color=((0.85, 0.85, 0.85),),
                          diffuse_color=((0.25, 0.25, 0.25),),
                          specular_color=((0.0, 0.0, 0.0),))
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights,
                              blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    out_frames = []
    mesh_color = (0.80, 0.82, 0.90)
    for i, (v, f) in enumerate(zip(verts_list, faces_list)):
        verts_rgb = torch.tensor(mesh_color, device=device).expand_as(v)[None]
        mesh = Meshes(verts=[v], faces=[f], textures=TexturesVertex(verts_features=verts_rgb))
        img = renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
        img_uint8 = (img * 255).astype(np.uint8)  # RGB
        img_resized = resize_with_crop_np(img_uint8, target_W, target_H)
        out_frames.append(img_resized)
        if (i + 1) % 10 == 0:
            print(f"    rendered {i+1}/{len(verts_list)}")
    return out_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default="./data/gt_data/dna/0188_02/smpl_mesh")
    ap.add_argument("--cam_pkl", default="./data/gt_data/dna/0188_02/cam.pkl")
    ap.add_argument("--out_dir", default="./output/0188_02_smpl_per_cam")
    ap.add_argument("--target_W", type=int, default=512)
    ap.add_argument("--target_H", type=int, default=768)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--cams", nargs="+",
                    default=["cam_00", "cam_03", "cam_07", "cam_10", "cam_13"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"loading meshes from {args.mesh_dir}")
    files, verts_list, faces_list = load_meshes(args.mesh_dir, device=device)
    print(f"  {len(files)} meshes loaded")

    with open(args.cam_pkl, "rb") as f:
        cams = pickle.load(f)

    for cid in args.cams:
        c = cams[cid]
        K = np.asarray(c["K"], dtype=np.float64)
        w2c = np.asarray(c["w2c"], dtype=np.float64)
        # Inferred render resolution from K principal point
        render_W = int(round(K[0, 2] * 2))
        render_H = int(round(K[1, 2] * 2))
        print(f"\n[{cid}] render at {render_W}x{render_H} -> crop+resize to {args.target_W}x{args.target_H}")

        frames = render_for_cam(verts_list, faces_list, K, w2c, render_H, render_W,
                                target_W=args.target_W, target_H=args.target_H, device=device)

        out_path = osp.join(args.out_dir, f"smpl_{cid}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                  (args.target_W, args.target_H))
        for f_rgb in frames:
            writer.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  saved {out_path} ({len(frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
