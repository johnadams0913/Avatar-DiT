"""
Build the dna_20case training dataset directory with symlinks and json index.

Output structure (under ./output/dna_train/):
  video/<case>_cam_XX.mp4   -> dna_gt_videos/<case>/gt_cam_XX.mp4
  mesh/<case>_cam_XX.mp4    -> dna_smpl_renders/<case>/smpl_cam_XX.mp4
  dna_train.json
"""
import json
import os
import os.path as osp


CASES = ["0012_09", "0019_06", "0025_11", "0034_04", "0094_02",
         "0124_03", "0152_01", "0165_08", "0188_02", "0219_07"]
DSET = "./output/dna_train"
GT_VIDEOS = "./output/dna_gt_videos"
SMPL_RENDERS = "./output/dna_smpl_renders"

video_dir = osp.join(DSET, "video")
mesh_dir = osp.join(DSET, "mesh")
os.makedirs(video_dir, exist_ok=True)
os.makedirs(mesh_dir, exist_ok=True)

entries = []
missing = []
for case_id in CASES:
    for cid in range(16):
        sample_name = f"{case_id}_cam_{cid:02d}"
        gt_src = osp.join(GT_VIDEOS, case_id, f"gt_cam_{cid:02d}.mp4")
        mesh_src = osp.join(SMPL_RENDERS, case_id, f"smpl_cam_{cid:02d}.mp4")
        gt_dst = osp.join(video_dir, f"{sample_name}.mp4")
        mesh_dst = osp.join(mesh_dir, f"{sample_name}.mp4")
        if not osp.exists(gt_src) or not osp.exists(mesh_src):
            missing.append((case_id, cid, osp.exists(gt_src), osp.exists(mesh_src)))
            continue
        if not osp.exists(gt_dst):
            os.symlink(gt_src, gt_dst)
        if not osp.exists(mesh_dst):
            os.symlink(mesh_src, mesh_dst)
        entries.append({
            "path": f"video/{sample_name}.mp4",
            "height": 768, "width": 512,
            "frame_number": 49,
            "tracking": False, "caption": "", "reference": None,
        })

json_path = osp.join(DSET, "dna_train.json")
with open(json_path, "w") as f:
    json.dump(entries, f, indent=2)
print(f"Built {len(entries)} samples; {len(missing)} missing.")
if missing:
    for m in missing[:10]:
        print(f"  missing: {m}")
