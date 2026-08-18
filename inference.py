import json
import os
import os.path as osp
import argparse
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

from backend.functool import save_video
from ckpt_util import load_config
from model.util import instantiate_from_config, default
from data.utils import normalize, to_tensor
from data.videoloader import (
    read_video_frames,
    frames_to_pil_images,
    resize_with_crop,
    resize_without_ratio,
)

# ── Dataset presets ──────────────────────────────────────────────────────────
DATASET_ROOT = os.environ.get("DATASET_ROOT", "./datasets")
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

DATASET_PRESETS = {
    "dna": {
        "dataroot": osp.join(DATASET_ROOT, "dna_finished/validation"),
        "camera_mode": "per_video_json",   # camera/{name}.json
    },
    "bili": {
        "dataroot": osp.join(DATASET_ROOT, "bili_data/validation"),
        "camera_mode": "per_video_npz",    # camera/{name}.npz
    },
    "mocap": {
        "dataroot": osp.join(DATASET_ROOT, "mocap_reorganized"),
        "camera_mode": "per_subject_json", # camera/{subject}.json -> [cam_id]
    },
}

MODEL_CONFIG_PATH = osp.join(PROJECT_ROOT, "configs/inference/multiview.yaml")
PRETRAINED_CKPT_PATH = osp.join(PROJECT_ROOT, "models/mv-new.safetensors")
OUTPUT_DIR = osp.join(PROJECT_ROOT, "validation_result")
FRAME_LENGTH = 81
LOAD_SIZE = (512, 768)
RESIZE_FN = "crop"
VIDEO_FORMAT = "mp4"
START_FRAME = 0
REFERENCE_FRAME_INDEX = 0
TEXT_PROMPT = "talking head"
STEPS = 20
GUIDANCE_SCALE = 1.0
FPS = 30
BATCH_SIZE = 1
NUM_THREADS = 4
EVAL_LOAD_SIZE = None
INTERPOLATE_POSITIONAL_EMBEDDING = False
SAVE_INPUT = False
SKIP_MISSING = True
IGNORE_KEYS = []


def load_camera_params_from_file(camera_path: str) -> torch.Tensor:
    with open(camera_path, "r", encoding="utf-8") as f:
        camera_data = json.load(f)

    intrinsic = camera_data.get("intrinsic")
    extrinsic = camera_data.get("extrinsic")
    distortion = camera_data.get("distortion", camera_data.get("D", [0.0, 0.0, 0.0, 0.0, 0.0]))
    scale = camera_data.get("scale", 1.0)

    if intrinsic is None or extrinsic is None:
        raise ValueError(f"Missing intrinsic/extrinsic in {camera_path}")

    if len(distortion) < 5:
        distortion = list(distortion) + [0.0] * (5 - len(distortion))

    extrinsic_tensor = torch.tensor(extrinsic).reshape(4, 4)
    intrinsic_tensor = torch.tensor(intrinsic).reshape(3, 3)

    k1, k2, p1, p2, k3 = distortion
    radial = 1.0 + k1 + k2 + k3
    intrinsic_tensor[0, 0] *= radial
    intrinsic_tensor[1, 1] *= radial
    intrinsic_tensor[0, 2] += p1 * intrinsic_tensor[0, 0]
    intrinsic_tensor[1, 2] += p2 * intrinsic_tensor[1, 1]

    intrinsic_tensor[:2] *= scale
    extrinsic_tensor[:3, 3] *= scale

    return torch.cat(
        [extrinsic_tensor.reshape(-1, 16), intrinsic_tensor.reshape(-1, 9)],
        dim=-1,
    )


def load_camera_params_from_npz(camera_path: str, frame_index: int = 0) -> torch.Tensor:
    """Load camera from NPZ (bili format): extrinsics (N,4,4), intrinsics (N,3,3)."""
    data = np.load(camera_path)
    extrinsic = torch.tensor(data["extrinsics"][frame_index], dtype=torch.float32).reshape(4, 4)
    intrinsic = torch.tensor(data["intrinsics"][frame_index], dtype=torch.float32).reshape(3, 3)
    return torch.cat(
        [extrinsic.reshape(-1, 16), intrinsic.reshape(-1, 9)],
        dim=-1,
    )


def load_camera_params_from_subject_json(camera_path: str, cam_id: str) -> torch.Tensor:
    """Load camera from per-subject JSON (mocap format): {subject}.json -> {cam_id}."""
    with open(camera_path, "r", encoding="utf-8") as f:
        camera_data = json.load(f)
    cam_info = camera_data[cam_id]
    scale = cam_info.get("scale", 1.0)
    distortion = cam_info.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
    if len(distortion) < 5:
        distortion = list(distortion) + [0.0] * (5 - len(distortion))

    extrinsic_tensor = torch.tensor(cam_info["extrinsic"], dtype=torch.float32).reshape(4, 4)
    intrinsic_tensor = torch.tensor(cam_info["intrinsic"], dtype=torch.float32).reshape(3, 3)

    k1, k2, p1, p2, k3 = distortion
    radial = 1.0 + k1 + k2 + k3
    intrinsic_tensor[0, 0] *= radial
    intrinsic_tensor[1, 1] *= radial
    intrinsic_tensor[0, 2] += p1 * intrinsic_tensor[0, 0]
    intrinsic_tensor[1, 2] += p2 * intrinsic_tensor[1, 1]

    intrinsic_tensor[:2] *= scale
    extrinsic_tensor[:3, 3] *= scale

    return torch.cat(
        [extrinsic_tensor.reshape(-1, 16), intrinsic_tensor.reshape(-1, 9)],
        dim=-1,
    )


class ValidationMultiViewDataset(data.Dataset):
    def __init__(
        self,
        dataroot: str,
        anchor_key: str,
        pose_key: str,
        camera_key: str,
        frame_length: int,
        load_size: tuple,
        resize_fn: str,
        camera_mode: str = "per_video_json",
        target_ids: Optional[List[str]] = None,
        target_views: Optional[List[str]] = None,
        reference_lookup: Optional[Dict[str, str]] = None,
        default_reference_view: Optional[str] = None,
        reference_frame_index: int = 0,
        self_reference: bool = False,
        video_format: str = "mp4",
        text_prompt: str = "talking head",
        start_frame: int = 0,
        skip_missing: bool = True,
    ):
        super().__init__()
        self.dataroot = dataroot
        self.anchor_key = anchor_key
        self.pose_key = pose_key
        self.camera_key = camera_key
        self.frame_length = frame_length
        self.load_size = load_size
        self.resize_fn = resize_fn
        self.camera_mode = camera_mode
        self.target_ids = set(target_ids or [])
        self.target_views = set(target_views or [])
        self.reference_lookup = reference_lookup or {}
        self.default_reference_view = default_reference_view
        self.reference_frame_index = reference_frame_index
        self.self_reference = self_reference
        self.video_format = video_format
        self.text_prompt = text_prompt
        self.start_frame = start_frame
        self.skip_missing = skip_missing

        self.video_dir = osp.join(dataroot, anchor_key)
        self.pose_dir = osp.join(dataroot, pose_key)
        self.camera_dir = osp.join(dataroot, camera_key)

        self.file_list = self._build_file_list()
        if not self.file_list:
            raise ValueError("No validation samples found. Check filters and paths.")

    def _resolve_camera_path(self, name):
        """Resolve camera file path based on camera_mode."""
        if self.camera_mode == "per_video_json":
            return osp.join(self.camera_dir, f"{name}.json")
        elif self.camera_mode == "per_video_npz":
            return osp.join(self.camera_dir, f"{name}.npz")
        elif self.camera_mode == "per_subject_json":
            # name = "{subject}_{cam_id}", e.g. "313_cam01"
            subject = name.split("_")[0]
            return osp.join(self.camera_dir, f"{subject}.json")
        return osp.join(self.camera_dir, f"{name}.json")

    def _build_file_list(self) -> List[dict]:
        file_list = []
        for fname in sorted(os.listdir(self.video_dir)):
            if not fname.endswith(f".{self.video_format}"):
                continue
            name = osp.splitext(fname)[0]

            # Parse base_id / view_id if underscore present
            if "_" in name:
                base_id, view_id = name.rsplit("_", 1)
            else:
                base_id, view_id = name, ""

            if self.target_ids and base_id not in self.target_ids:
                continue
            if self.target_views and view_id not in self.target_views:
                continue

            video_path = osp.join(self.video_dir, fname)
            pose_path = osp.join(self.pose_dir, fname)
            camera_path = self._resolve_camera_path(name)

            if not osp.isfile(pose_path):
                if self.skip_missing:
                    continue
                raise ValueError(f"Missing mesh for {name}")
            if not osp.isfile(camera_path):
                if self.skip_missing:
                    continue
                raise ValueError(f"Missing camera for {name}")

            # Self-reference mode: reference = input video itself
            if self.self_reference:
                reference_video_path = video_path
            else:
                ref_view = self.reference_lookup.get(base_id, self.default_reference_view)
                if not ref_view:
                    if self.skip_missing:
                        continue
                    raise ValueError(f"Missing reference view for {base_id}")
                candidate = osp.join(self.video_dir, f"{base_id}_{ref_view}.{self.video_format}")
                if not osp.isfile(candidate):
                    if self.skip_missing:
                        continue
                    raise ValueError(f"Missing reference video {candidate}")
                reference_video_path = candidate

            file_list.append(
                {
                    "name": name,
                    "base_id": base_id,
                    "view_id": view_id,
                    "video_path": video_path,
                    "pose_path": pose_path,
                    "camera_path": camera_path,
                    "reference_video_path": reference_video_path,
                    "reference_name": name if self.self_reference else osp.splitext(osp.basename(reference_video_path))[0],
                }
            )
        return file_list

    def _get_resize_func(self):
        if self.resize_fn == "crop":
            return partial(resize_with_crop, new_size=self.load_size)
        return partial(resize_without_ratio, new_size=self.load_size)

    def _load_reference_image(self, item, frames):
        if self.self_reference:
            # Use first frame from the already-loaded input frames
            return frames[self.reference_frame_index]
        if not item["reference_video_path"]:
            raise ValueError(f"Missing reference video for {item['base_id']}")
        ref_frames, _ = read_video_frames(
            item["reference_video_path"], 1, self.reference_frame_index
        )
        return frames_to_pil_images(ref_frames)[0]

    def _load_camera(self, item) -> torch.Tensor:
        if self.camera_mode == "per_video_json":
            return load_camera_params_from_file(item["camera_path"])
        elif self.camera_mode == "per_video_npz":
            return load_camera_params_from_npz(item["camera_path"], frame_index=0)
        elif self.camera_mode == "per_subject_json":
            cam_id = item["name"].split("_", 1)[1]  # e.g. "313_cam01" -> "cam01"
            return load_camera_params_from_subject_json(item["camera_path"], cam_id)
        return load_camera_params_from_file(item["camera_path"])

    def __getitem__(self, index):
        item = self.file_list[index]
        resize_func = self._get_resize_func()

        frames, _ = read_video_frames(item["video_path"], self.frame_length, self.start_frame)
        pose_frames, _ = read_video_frames(item["pose_path"], self.frame_length, self.start_frame)

        frames = frames_to_pil_images(frames)
        pose_frames = frames_to_pil_images(pose_frames)

        actual_len = min(self.frame_length, len(frames), len(pose_frames))
        img_chunk = []
        pose_chunk = []
        for idx in range(actual_len):
            img = normalize(resize_func(frames[idx]))
            pose = to_tensor(resize_func(pose_frames[idx]))
            img_chunk.append(img)
            pose_chunk.append(pose)

        reference_img = self._load_reference_image(item, frames)
        reference_img = normalize(resize_func(reference_img))

        camera = self._load_camera(item)

        data_item = {
            "image": torch.stack(img_chunk, dim=0),
            "pose": torch.stack(pose_chunk, dim=0),
            "reference": torch.stack([reference_img], dim=0),
            "text": self.text_prompt,
            "camera": camera,
            "name": item["name"],
            "base_id": item["base_id"],
            "view_id": item["view_id"],
            "reference_name": item["reference_name"],
        }
        return data_item

    def __len__(self):
        return len(self.file_list)


def modify_z_shape(params, eval_size, interpolate=False):
    ldm_image_size = params.image_size
    first_model_config = params.first_stage_config
    first_stage_image_size = first_model_config.params.ddconfig.resolution
    scale_factor = first_stage_image_size // ldm_image_size

    params.image_size = eval_size // scale_factor
    if interpolate:
        try:
            params.cond_stage_config.params.clip_config.scale_factor = eval_size / first_stage_image_size
        except:
            params.cond_stage_config.params.scale_factor = eval_size / first_stage_image_size
    return params


def resolve_eval_load_size(dataloader_params, opt_eval_load_size):
    if opt_eval_load_size is None:
        return tuple(dataloader_params.get("load_size", (512, 768)))
    return (opt_eval_load_size, opt_eval_load_size)


def tensor_video_to_uint8(frames: torch.Tensor, normalized: bool) -> torch.Tensor:
    if normalized:
        frames = (frames.clamp(-1, 1) + 1.0) / 2.0
    else:
        frames = frames.clamp(0, 1)
    frames = (frames * 255.0).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).cpu().numpy()


def split_file_list(file_list: List[dict], num_chunks: int, chunk_id: int) -> List[dict]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError("chunk_id must be in [0, num_chunks).")
    total = len(file_list)
    start = (total * chunk_id) // num_chunks
    end = (total * (chunk_id + 1)) // num_chunks
    return file_list[start:end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        choices=list(DATASET_PRESETS.keys()),
                        help="Dataset preset: dna, bili, mocap")
    parser.add_argument("--num-chunks", "-n", type=int, default=1)
    parser.add_argument("--chunk-id", "-nid", type=int, default=0)
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    data_root = preset["dataroot"]
    camera_mode = preset["camera_mode"]
    out_root = osp.join(OUTPUT_DIR, args.dataset)

    configs = load_config(default(MODEL_CONFIG_PATH, MODEL_CONFIG_PATH))

    model_params = configs.model.params
    if EVAL_LOAD_SIZE is not None and "image_size" in model_params:
        configs.model.params = modify_z_shape(
            model_params, EVAL_LOAD_SIZE, INTERPOLATE_POSITIONAL_EMBEDDING
        )

    os.makedirs(out_root, exist_ok=True)
    output_dir = osp.join(out_root, "output")
    input_dir = osp.join(out_root, "input")
    pose_dir = osp.join(out_root, "mesh")
    reference_dir = osp.join(out_root, "reference")
    for d in [output_dir, input_dir, pose_dir, reference_dir]:
        os.makedirs(d, exist_ok=True)

    model = instantiate_from_config(configs.model).cuda().eval()
    model.switch_to_fp16()
    if PRETRAINED_CKPT_PATH:
        model.init_from_ckpt(PRETRAINED_CKPT_PATH, ignore_keys=IGNORE_KEYS)

    dataloader_params = getattr(configs, "dataloader", {}).get("params", {})
    frame_length = dataloader_params.get("frame_length", FRAME_LENGTH)
    anchor_key = dataloader_params.get("anchor_key", "video")
    pose_key = dataloader_params.get("pose_key", "mesh")
    camera_key = dataloader_params.get("camera_key", "camera")
    resize_fn = dataloader_params.get("resize_fn", RESIZE_FN)
    load_size = resolve_eval_load_size(dataloader_params, EVAL_LOAD_SIZE)

    dataset = ValidationMultiViewDataset(
        dataroot=data_root,
        anchor_key=anchor_key,
        pose_key=pose_key,
        camera_key=camera_key,
        frame_length=frame_length,
        load_size=LOAD_SIZE if EVAL_LOAD_SIZE is None else load_size,
        resize_fn=resize_fn,
        camera_mode=camera_mode,
        self_reference=True,
        reference_frame_index=REFERENCE_FRAME_INDEX,
        video_format=VIDEO_FORMAT,
        text_prompt=TEXT_PROMPT,
        start_frame=START_FRAME,
        skip_missing=SKIP_MISSING,
    )
    print(f"[{args.dataset}] Found {len(dataset.file_list)} samples.")
    dataset.file_list = split_file_list(dataset.file_list, args.num_chunks, args.chunk_id)
    print(f"[{args.dataset}] Chunk {args.chunk_id}/{args.num_chunks}: {len(dataset.file_list)} samples.")
    if not dataset.file_list:
        raise ValueError("No validation samples found for selected chunk.")

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_THREADS,
        pin_memory=True,
        drop_last=False,
    )

    model.training = False
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation [{args.dataset}]"):
            batch_size = batch["image"].shape[0]
            for idx in range(batch_size):
                sample_id = batch["name"][idx]
                output_path = osp.join(output_dir, f"{sample_id}.mp4")
                if osp.isfile(output_path):
                    continue

                image = batch["image"][idx]
                pose = batch["pose"][idx]
                reference = batch["reference"][idx]
                camera = batch["camera"][idx]
                text_prompt = batch["text"][idx]

                output_frames = model.generate_video(
                    control=pose,
                    reference=reference,
                    prompt=text_prompt,
                    negative_prompt="",
                    video=None,
                    face=None,
                    flame=None,
                    camera=camera,
                    smpl=None,
                    gs=GUIDANCE_SCALE,
                    height=LOAD_SIZE[1],
                    width=LOAD_SIZE[0],
                    step=STEPS,
                    low_vram=False,
                    sampler="unipc",
                    frame_length=frame_length,
                    chunk_num=1,
                    seed=42,
                    motion_settings=dict(fps=FPS),
                )

                output_video = tensor_video_to_uint8(output_frames, normalized=True)
                input_video = tensor_video_to_uint8(image, normalized=True)
                pose_video = tensor_video_to_uint8(pose, normalized=False)
                reference_image = tensor_video_to_uint8(reference, normalized=True)[0]

                save_video(output_video, output_path, fps=FPS)
                save_video(input_video, osp.join(input_dir, f"{sample_id}.mp4"), fps=FPS)
                save_video(pose_video, osp.join(pose_dir, f"{sample_id}.mp4"), fps=FPS)
                Image.fromarray(reference_image).save(
                    osp.join(reference_dir, f"{sample_id}.png")
                )