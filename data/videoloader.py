import os
import os.path as osp
import json
import random
import cv2
import traceback
import pandas as pd

from .utils import *
from functools import partial
from typing import List, Union, Tuple
from PIL import ImageFile
from tqdm import tqdm

import torch
import torch.utils.data as data


DEBUG = False
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'm4v'}
ImageFile.LOAD_TRUNCATED_IMAGES = True
bic = transforms.InterpolationMode.BICUBIC


def read_video_frames(video_path: str, num_frames: int, start_frame: int = 0) -> Tuple[List[np.ndarray], float]:
    """
    Read frames from video file starting from start_frame.

    Args:
        video_path: Path to video file
        start_frame: Starting frame index (0-based)
        num_frames: Number of frames to read. If None, read all frames from start_frame

    Returns:
        List of frames as numpy arrays in RGB format, and fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if start_frame >= total_frames:
        raise ValueError(f"Video {video_path} start frame {start_frame} is out of range. Total frames: {total_frames}")

    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    read_count = 0

    while read_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        read_count += 1

    cap.release()
    return frames, fps


def frames_to_pil_images(frames: List[np.ndarray]) -> List[Image.Image]:
    """Convert list of numpy frames to PIL Images."""
    return [Image.fromarray(frame) for frame in frames]


def resize_without_ratio(img, new_size):
    """Resize image to specific size without maintaining aspect ratio."""
    return transforms.Resize(new_size[::-1], bic)(img)


def resize_with_crop(img, new_size):
    w, h = img.size
    target_w, target_h = new_size

    # Calculate aspect ratios
    img_ratio = h / w
    target_ratio = target_h / target_w

    if img_ratio > target_ratio:
        # Original image is relatively taller, crop vertically (top and bottom)
        # Scale width to match target width, then crop height
        scale_factor = target_w / w
        scaled_h = int(h * scale_factor)
        crop_h = target_h

        # Calculate crop coordinates with height//2 as center
        center_y = scaled_h // 2
        start_y = max(0, center_y - crop_h // 2)
        end_y = min(scaled_h, start_y + crop_h)

        # First resize to match width, then crop height
        img = img.resize((target_w, scaled_h), Image.LANCZOS)
        img = img.crop((0, start_y, target_w, end_y))
    else:
        # Original image is relatively wider, crop horizontally (left and right)
        # Scale height to match target height, then crop width
        scale_factor = target_h / h
        scaled_w = int(w * scale_factor)
        crop_w = target_w

        # Calculate crop coordinates with width//2 as center
        center_x = scaled_w // 2
        start_x = max(0, center_x - crop_w // 2)
        end_x = min(scaled_w, start_x + crop_w)

        # First resize to match height, then crop width
        img = img.resize((scaled_w, target_h), Image.LANCZOS)
        img = img.crop((start_x, 0, end_x, target_h))

    return resize_without_ratio(img, new_size)


def pad_and_resize(image, target_size=(512, 768)):
    w, h = image.size
    ratio = target_size[1] / target_size[0]
    temp_size = (w, int(w * ratio))
    bg = Image.new('RGB', (temp_size), (255, 255, 255))
    x, y = map(lambda t: (temp_size[t] - image.size[t]) // 2, (0, 1))

    bg.paste(image, (x, y))
    resized = bg.resize(target_size, Image.BICUBIC)
    return resized


class VideoMultiModalDataset(data.Dataset):
    """
    Video-based multimodal dataset that reads videos directly instead of frame-by-frame.
    Supports both directory-based discovery and JSON-based configuration.

    Standard data structure:
    Root path
        |---video (or custom anchor_key)
            |---0001.mp4
            |---0002.mp4
            ...
        |---pose (or custom pose_key)
            |---0001.mp4
            |---0002.mp4
            ...
        |---mask (optional)
        |---depth (optional)
        |---reference.png (optional)
        |---dataset.json (optional, for JSON-based configuration)

    Args:
        load_size: Single tuple (w, h) or list of tuples for random load size per batch.
                   When a list is provided, each batch will randomly select one size.
    """

    def __init__(
            self,
            dataroot,
            anchor_key="video",
            pose_key="pose",
            face_key=None,
            depth_key=None,
            mask_key=None,
            flame_key=None,
            track_key="track_boxes",
            frame_length=81,
            load_size=(512, 768),
            video_format="mp4",
            ref_key=None,
            mask_expansion_size=(30, 20),
            mask_expansion_p=0.5,
            json_key=None,
            resize_fn="crop",
            use_first_frame_reference=False,
            lora_key=None,
            context_overlap=0,
            candidate_sizes=None,
            *args,
            **kwargs
    ):
        super().__init__()
        self.dataroot = dataroot
        self.pose_key = pose_key
        self.anchor_key = anchor_key
        self.face_key = face_key
        self.mask_key = mask_key
        self.depth_key = depth_key
        self.flame_key = flame_key
        self.track_key = track_key
        self.ref_key = ref_key
        self.json_key = json_key
        self.lora_key = lora_key

        self.tracking = exists(track_key)
        self.face_guided = exists(face_key)
        self.mask_guided = exists(mask_key)
        self.depth_guided = exists(depth_key)
        self.flame_guided = exists(flame_key)

        self.mask_expansion_size = mask_expansion_size
        self.mask_expansion_p = mask_expansion_p
        self.frame_length = frame_length
        self.context_overlap = context_overlap
        self.video_format = video_format
        self.use_first_frame_reference = use_first_frame_reference
        self.resize_fn_type = resize_fn
        self.resize_fn = resize_with_crop if resize_fn == "crop" else resize_without_ratio

        # Candidate sizes: ordered list of (w, h) to try based on video native resolution.
        # The first candidate whose dimensions are both <= the video's native resolution is used.
        # If none match, the video is skipped.
        # e.g., candidate_sizes=[[720, 1280], [1280, 720], [1024, 1024]]
        self.candidate_sizes = [tuple(s) for s in candidate_sizes] if candidate_sizes is not None else None

        # Support random load size from a list
        if isinstance(load_size, list):
            self.load_size_list = load_size
            self.load_size = load_size[0]  # Default to first size
        else:
            self.load_size_list = None
            self.load_size = load_size

        self.prepare_file_list()

    def sample_load_size(self):
        """Randomly sample a load size from the list if available."""
        if self.load_size_list is not None:
            return self.load_size_list[random.randint(len(self.load_size_list))]
        return self.load_size

    def adapt_load_size(self, video_path: str, target_size: tuple):
        """Select load size based on video's native resolution.

        If candidate_sizes is set, try each candidate in order and return the
        first one where vid_w >= candidate_w and vid_h >= candidate_h.
        Returns None if no candidate fits (caller should skip this video).
        """
        if self.candidate_sizes is None:
            return target_size
        cap = cv2.VideoCapture(video_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        for cw, ch in self.candidate_sizes:
            if vid_w >= cw and vid_h >= ch:
                return (cw, ch)
        return None

    def prepare_file_list(self):
        json_path = osp.join(self.dataroot, f"{self.json_key}.json")
        with open(json_path, 'r') as f:
            video_dict = json.load(f)
        file_list = []
        skipped = 0
        for vd in video_dict:
            video_path = osp.join(self.dataroot, vd["path"])
            vid_name = osp.splitext(osp.basename(video_path))[0]
            total_frames = int(vd["frame_number"])

            # Filter by resolution at init time
            matched_size = self.adapt_load_size(video_path, self.load_size)
            if matched_size is None:
                skipped += 1
                continue

            for start_idx in range(
                0, total_frames - self.frame_length + 1, self.frame_length
            ):
                chunk_info = {
                    'video_name': vid_name,
                    'video_path': video_path,
                    'video_length': total_frames,
                    'start_frame': start_idx,
                    'tracking': vd.get("tracking", False),
                    'caption': vd.get("caption", None),
                    'reference': vd.get("reference", None),
                    'matched_size': matched_size,
                }
                file_list.append(chunk_info)
        if skipped > 0:
            print(f"[VideoLoader] Skipped {skipped} videos due to resolution too low for any candidate size")
        self.file_list = file_list

    def load_motion_code(self, video_path, start_frame, frame_length):
        flame_path = video_path.replace(f"/{self.anchor_key}/", f"/{self.flame_key}/")
        flame_path = osp.splitext(flame_path)[0] + ".parquet"
        df = pd.read_parquet(flame_path)
        motion_data = df.values.astype(np.float16)
        motion_chunk = motion_data[start_frame: start_frame + frame_length]
        if len(motion_chunk) != frame_length:
            raise ValueError(
                f"Flame frame count mismatch: {flame_path}, "
                f"expected {frame_length}, got {len(motion_chunk)} "
                f"(total: {len(motion_data)}, start: {start_frame})"
            )
        return motion_chunk

    def get_reference_info(self, video_path) -> Union[int, str]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        ref_frame_idx = random.randint(0, max(1, total_frames))
        return ref_frame_idx

    def load_video_frames(self, video_path: str, start_frame: int, num_frames: int, convert_mode: str = "RGB") -> Tuple[List[Image.Image], float]:
        """Load frames from video and convert to specified mode."""
        frames, fps = read_video_frames(video_path, num_frames, start_frame)
        if len(frames) != num_frames:
            raise ValueError(
                f"Video frame count mismatch: {video_path}, "
                f"expected {num_frames}, got {len(frames)} "
                f"(start: {start_frame})"
            )
        pil_frames = frames_to_pil_images(frames)

        if convert_mode != "RGB":
            return [frame.convert(convert_mode) for frame in pil_frames], fps
        return pil_frames, fps

    def load_modality_data(self, video_path: str, start_frame: int, num_frames: int, modality: str) -> List[Image.Image]:
        """Load data for specified modality (pose, mask, depth)."""
        modality_map = {
            "pose": (self.pose_key, "RGB"),
            "face": (self.face_key, "RGB"),
            "mask": (self.mask_key, "L"),
            "depth": (self.depth_key, "L")
        }

        assert modality in modality_map, f"Modality {modality} not in {modality_map.keys()}"
        modality_key, convert_mode = modality_map[modality]
        modality_video_path = video_path.replace(f"/{self.anchor_key}/", f"/{modality_key}/")
        frames, _ = self.load_video_frames(modality_video_path, start_frame, num_frames, convert_mode)
        if len(frames) != num_frames:
            raise ValueError(
                f"{modality.capitalize()} frame count mismatch: {modality_video_path}, "
                f"expected {num_frames}, got {len(frames)}"
            )
        return frames

    def preprocess_frame(
            self,
            image: Image.Image,
            pose: Image.Image = None,
            face: Image.Image = None,
            mask: Image.Image = None,
            depth: Image.Image = None,
            resize_func=None
    ):
        """Preprocess individual frame data."""
        if resize_func:
            image = resize_func(image)
            if exists(pose):
                pose = resize_func(pose)

        image = normalize(image)
        if exists(pose):
            pose = to_tensor(pose)

        if exists(face):
            face = resize_without_ratio(face, (512, 512))
            face = to_tensor(face)

        if exists(mask):
            mask = resize_func(mask)
            mask = to_tensor(mask)
            mask[mask > 0] = 1
            if self.mask_expansion_p > 0 and random.rand() < self.mask_expansion_p:
                mask = mask_expansion(mask, *self.mask_expansion_size)
            else:
                mask = F.max_pool2d(mask, kernel_size=21, stride=1, padding=10)
            masked_image = torch.where(mask > 0, torch.ones_like(image), image)
        else:
            mask = None
            masked_image = None

        # Process depth
        if exists(depth):
            depth = resize_func(depth)
            depth = to_tensor(depth)

        return image, pose, face, mask, masked_image, depth

    def read_video_data(self, chunk_info: dict):
        """Read and process video data for given chunk."""
        vid_name = chunk_info['video_name']
        video_path = chunk_info['video_path']
        start_frame = chunk_info['start_frame']

        # Caption priority: JSON caption > track caption > lora_key > default
        caption = chunk_info.get('caption', None) or self.lora_key or ""

        # Use pre-matched size from init, or fall back to sample
        current_load_size = chunk_info.get('matched_size') or self.sample_load_size()
        frame_length = self.frame_length

        if self.flame_guided:
            motion_code = self.load_motion_code(video_path, start_frame, frame_length)

        center = None
        if chunk_info.get('tracking', False):
            track_path = video_path.replace(f"/{self.anchor_key}/", f"/{self.track_key}/")
            track_file = osp.join(osp.dirname(track_path), f"{vid_name}.json")
            if osp.isfile(track_file):
                with open(track_file, 'r') as f:
                    track_data = json.load(f)
                    center = track_data.get("mean_center", None)
                    track_caption = track_data.get("caption", None)
                    if track_caption:
                        caption = track_caption

        if exists(center):
            resize_func = partial(track_crop, mean_center=center, target_size=current_load_size)
        else:
            resize_func = partial(self.resize_fn, new_size=current_load_size)

        # Load main video frames
        frames, fps = self.load_video_frames(video_path, start_frame, frame_length)
        pose_frames = self.load_modality_data(video_path, start_frame, frame_length, "pose") if exists(self.pose_key) else None
        face_frames = self.load_modality_data(video_path, start_frame, frame_length, "face") if self.face_guided else None
        mask_frames = self.load_modality_data(video_path, start_frame, frame_length, "mask") if self.mask_guided else None
        depth_frames = self.load_modality_data(video_path, start_frame, frame_length, "depth") if self.depth_guided else None

        if len(frames) != frame_length:
            raise ValueError(
                f"Frame count validation failed for {vid_name}: "
                f"video={len(frames)}, expected={frame_length}"
            )
        if exists(pose_frames) and len(pose_frames) != frame_length:
            raise ValueError(
                f"Pose frame count validation failed for {vid_name}: "
                f"pose={len(pose_frames)}, expected={frame_length}"
            )

        # Load reference image
        ref_path = chunk_info.get('reference', None)
        if exists(ref_path):
            reference = Image.open(osp.join(self.dataroot, chunk_info['reference'])).convert("RGB")
        else:
            if self.use_first_frame_reference:
                reference = frames[0]
            else:
                ref_frames, _ = read_video_frames(
                    chunk_info['video_path'], 1, random.randint(chunk_info['video_length'])
                )
                reference = frames_to_pil_images(ref_frames)[0]

        reference = resize_func(reference)
        reference = normalize(reference)

        # Process all frames (multi-frame mode)
        img_chunk = []
        pose_chunk = []
        face_chunk = []
        mask_chunk = []
        masked_chunk = []
        depth_chunk = []

        for i in range(frame_length):
            frame_pose, frame_face, frame_mask, frame_depth = map(
                lambda t: t[i] if t else None,
                (pose_frames, face_frames, mask_frames, depth_frames)
            )
            img, p, f, m, mi, d = self.preprocess_frame(
                frames[i], frame_pose, frame_face, frame_mask, frame_depth, resize_func
            )

            img_chunk.append(img)
            if exists(p):
                pose_chunk.append(p)

            if self.face_guided:
                face_chunk.append(f)

            if self.mask_guided:
                mask_chunk.append(m)
                masked_chunk.append(mi)

            if self.depth_guided:
                depth_chunk.append(d)

        # Load context (full previous chunk) for long video continuation training
        # 70% of the time, load the previous chunk as context; 30% simulate first chunk
        ctx_chunk = None
        if self.context_overlap > 0:
            chunk1_start = start_frame - (frame_length - self.context_overlap)
            if chunk1_start >= 0 and random.random() >= 0.3:
                chunk1_frames, _ = self.load_video_frames(video_path, chunk1_start, frame_length)
                ctx_chunk = [normalize(resize_func(f)) for f in chunk1_frames]

        # Stack tensors
        data_item = {
            "image": torch.stack(img_chunk, dim=0),
            "text": caption,
            "reference": torch.stack([reference], dim=0),
            "fps": fps,
            "load_size": current_load_size,  # Include current load size in output
        }

        if pose_chunk:
            data_item["pose"] = torch.stack(pose_chunk, dim=0)

        if ctx_chunk is not None:
            data_item["context"] = torch.stack(ctx_chunk, dim=0)

        if self.face_guided:
            data_item.update({
                "face": torch.stack(face_chunk, dim=0)
            })
        if self.mask_guided:
            data_item.update({
                "mask": torch.stack(mask_chunk, dim=0),
                "masked_image": torch.stack(masked_chunk, dim=0),
            })

        if self.depth_guided:
            data_item.update({
                "depth": torch.stack(depth_chunk, dim=0),
            })

        if self.flame_guided:
            data_item.update({
                "motion_code": torch.tensor(motion_code)
            })

        return data_item

    def __getitem__(self, index):
        if DEBUG:
            chunk_info = self.file_list[index]
            return self.read_video_data(chunk_info)
        else:
            while True:
                try:
                    chunk_info = self.file_list[index]
                    return self.read_video_data(chunk_info)
                except Exception as e:
                    error_msg = str(e)
                    if "frame count" in error_msg.lower() or "mismatch" in error_msg.lower():
                        tqdm.write(f"Frame count error for {self.file_list[index]['video_path']}: {error_msg}")
                    else:
                        tqdm.write(f"Cannot load video {self.file_list[index]['video_path']}")
                        tqdm.write(traceback.format_exc())
                    index = (index + 1) % len(self.file_list)

    def __len__(self):
        return len(self.file_list)


class MultiViewVideoDataset(VideoMultiModalDataset):
    def __init__(self, camera_key="camera", max_reference_num=1, relative_camera=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_key = camera_key
        self.camera_guided = exists(camera_key)
        self.max_reference_num = max_reference_num
        self.relative_camera = relative_camera

    def _collect_other_view_videos(self, video_path: str, base_id: str, camera_id: str) -> List[str]:
        video_dir = osp.dirname(video_path)
        other_videos = []
        for fname in os.listdir(video_dir):
            name, ext = fname.rsplit(".", 1)
            vid_base, vid_cam = name.rsplit("_", 1)
            if vid_base == base_id and vid_cam != camera_id:
                other_videos.append(osp.join(video_dir, fname))
        return other_videos

    def _sample_reference_frame(self, ref_video_path: str) -> Image.Image:
        cap = cv2.VideoCapture(ref_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            raise ValueError(f"Video {ref_video_path} has no frames")
        start_frame = random.randint(0, total_frames - 1)
        ref_frames, _ = read_video_frames(ref_video_path, 1, start_frame)
        return frames_to_pil_images(ref_frames)[0]

    def load_camera_params(self, video_path: str, base_id: str, camera_id: str):
        camera_dir = osp.dirname(video_path.replace(f"/{self.anchor_key}/", f"/{self.camera_key}/"))
        camera_path = osp.join(camera_dir, f"{base_id}.json")
        if not osp.exists(camera_path):
            return None
        with open(camera_path, "r", encoding="utf-8") as f:
            camera_data = json.load(f)
        view_data = camera_data.get(camera_id) or camera_data.get(str(camera_id))
        if not view_data:
            return None

        extrinsic = view_data.get("extrinsic")
        intrinsic = view_data.get("intrinsic")
        distortion = view_data.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
        scale = view_data.get("scale", 1.0)

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

        camera = torch.cat([extrinsic_tensor.reshape(-1, 16), intrinsic_tensor.reshape(-1, 9)], dim=-1)
        return camera

    def read_video_data(self, chunk_info: dict):
        data_item = super().read_video_data(chunk_info)
        if not self.camera_guided:
            return data_item

        vid_name = chunk_info["video_name"]
        video_path = chunk_info["video_path"]
        current_load_size = data_item.get("load_size", self.load_size)

        try:
            base_id, camera_id = vid_name.rsplit('_', 1)
        except:
            data_item["camera"] = torch.zeros(25, dtype=torch.float32)
            return data_item

        camera = self.load_camera_params(video_path, base_id, camera_id)
        data_item["camera"] = camera if exists(camera) else torch.zeros(25, dtype=torch.float32)

        other_videos = self._collect_other_view_videos(video_path, base_id, camera_id)
        if other_videos and self.max_reference_num > 0:
            max_num = min(self.max_reference_num, len(other_videos))
            ref_num = int(random.randint(1, max_num + 1))
            if ref_num == 1:
                chosen_videos = [random.choice(other_videos)]
            else:
                chosen_videos = random.choice(other_videos, size=ref_num, replace=False).tolist()

            center = None
            if chunk_info.get('tracking', False):
                track_path = video_path.replace(f"/{self.anchor_key}/", f"/{self.track_key}/")
                track_file = osp.join(osp.dirname(track_path), f"{vid_name}.json")
                with open(track_file, 'r') as f:
                    track_data = json.load(f)
                    center = track_data.get("mean_center", None)

            if exists(center):
                resize_func = partial(track_crop, mean_center=center, target_size=current_load_size)
            else:
                resize_func = partial(self.resize_fn, new_size=current_load_size)

            references = []
            for ref_video_path in chosen_videos:
                reference = self._sample_reference_frame(ref_video_path)
                reference = resize_func(reference)
                reference = normalize(reference)
                references.append(reference)

            if references:
                data_item["reference"] = torch.stack(references, dim=0)
        return data_item


class RandomSizeCollateFn:
    """
    Custom collate function that ensures all samples in a batch use the same load size.
    When load_size_list is provided, randomly selects one size for the entire batch.
    """
    def __init__(self, load_size_list=None):
        self.load_size_list = load_size_list

    def __call__(self, batch):
        # Standard collation: stack tensors, keep lists for strings
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], (str, int, float, tuple)):
                collated[key] = values
            else:
                collated[key] = values
        return collated


class PreEncodedDataset(data.Dataset):
    """Load pre-encoded VAE latents and T5 text embeddings from .pt files.

    Expected directory structure:
        {dataroot}/{encoded_dir}/
            latents/{vid_name}/chunk_{idx}_{W}x{H}.pt
            text_embeddings/{vid_name}.pt
            image_embeddings/{vid_name}/chunk_{idx}.pt  (optional, for I2V)

    Each latent .pt contains: video_latent, video_name, caption, ...
    Each text .pt contains: text_embedding, caption, video_name
    Each image .pt contains: image_embedding (CLIP), video_name, chunk_index

    Args:
        dataroot: Base dataset directory.
        encoded_dir: Subdirectory under dataroot containing encoded data.
        json_path: Path to JSON index file (relative to dataroot). If provided,
            only videos listed in the JSON are loaded. Each entry must have a
            "vid_name" field (or "path" from which vid_name is derived).
            If None, scans all videos in the latents directory.
    """

    def __init__(self, dataroot, text_len=512, load_image_embeddings=False,
                 max_latent_frames=None, encoded_dir="encoded_distill",
                 json_path=None, **kwargs):
        super().__init__()
        base = osp.join(dataroot, encoded_dir)
        self.latent_dir = osp.join(base, "latents")
        self.text_dir = osp.join(base, "text_embeddings")
        self.image_dir = osp.join(base, "image_embeddings")
        self.text_len = text_len
        self.load_image_embeddings = load_image_embeddings
        self.max_latent_frames = max_latent_frames

        # Determine which videos to include
        allowed_vids = None
        if json_path is not None:
            full_json = osp.join(dataroot, json_path) if not osp.isabs(json_path) else json_path
            with open(full_json) as f:
                index = json.load(f)
            allowed_vids = set()
            for item in index:
                if "vid_name" in item:
                    allowed_vids.add(item["vid_name"])
                elif "path" in item:
                    # Derive vid_name from path: "bili_data/video/0001.mp4" -> "0001"
                    allowed_vids.add(osp.splitext(osp.basename(item["path"]))[0])
            print(f"[PreEncodedDataset] JSON index: {full_json} ({len(allowed_vids)} videos)")

        # Scan latent .pt files, filtered by JSON if provided
        self.file_list = []
        for vid_name in sorted(os.listdir(self.latent_dir)):
            if allowed_vids is not None and vid_name not in allowed_vids:
                continue
            vid_dir = osp.join(self.latent_dir, vid_name)
            if not osp.isdir(vid_dir):
                continue
            for fname in sorted(os.listdir(vid_dir)):
                if fname.endswith(".pt"):
                    chunk_idx = int(fname.split("_")[1])
                    self.file_list.append((osp.join(vid_dir, fname), vid_name, chunk_idx))
        print(f"[PreEncodedDataset] Found {len(self.file_list)} pre-encoded chunks"
              f" from {base}/"
              f" (image_embeddings={'yes' if load_image_embeddings else 'no'})")

        # Cache text embeddings in memory (they're small: ~2K files × ~200KB each)
        self._text_cache = {}

    def _load_text_embedding(self, vid_name):
        if vid_name not in self._text_cache:
            text_path = osp.join(self.text_dir, f"{vid_name}.pt")
            text_data = torch.load(text_path, map_location="cpu", weights_only=True)
            self._text_cache[vid_name] = text_data["text_embedding"]
        return self._text_cache[vid_name]

    def _load_image_embedding(self, vid_name, chunk_idx):
        img_path = osp.join(self.image_dir, vid_name, f"chunk_{chunk_idx:04d}.pt")
        img_data = torch.load(img_path, map_location="cpu", weights_only=True)
        return img_data["image_embedding"]  # [257, 1280]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path, vid_name, chunk_idx = self.file_list[index]
        latent_data = torch.load(file_path, map_location="cpu",
                                 weights_only=True)
        video_latent = latent_data["video_latent"].float()  # [16, T, H, W]

        # Random temporal crop if max_latent_frames is set
        if self.max_latent_frames and video_latent.shape[1] > self.max_latent_frames:
            max_start = video_latent.shape[1] - self.max_latent_frames
            start = random.randint(0, max_start)
            video_latent = video_latent[:, start:start + self.max_latent_frames]

        caption = latent_data.get("caption", "")

        text_emb = self._load_text_embedding(vid_name).float()  # [seq_len, 4096]
        # Pad to fixed text_len for batching
        seq_len = text_emb.shape[0]
        if seq_len < self.text_len:
            pad = torch.zeros(self.text_len - seq_len, text_emb.shape[1])
            text_emb = torch.cat([text_emb, pad], dim=0)
        else:
            text_emb = text_emb[:self.text_len]

        result = {
            "latent": video_latent,
            "text_embedding": text_emb,
            "text_seq_len": min(seq_len, self.text_len),
            "text": caption,
        }

        if self.load_image_embeddings:
            img_emb = self._load_image_embedding(vid_name, chunk_idx).float()
            # ref_latent: first frame of video latent [C, 1, H, W]
            ref_latent = video_latent[:, :1, :, :]  # [16, 1, H, W]
            result["image_embedding"] = img_emb      # [257, 1280]
            result["ref_latent"] = ref_latent         # [16, 1, H, W]

        return result


class TrajectoryDataset(data.Dataset):
    """Load pre-computed ODE trajectories for ODE-Init training.

    Expected directory structure:
        {dataroot}/{traj_dir}/traj_00000.pt, traj_00001.pt, ...

    Each .pt contains:
        trajectory: [5, C, T, H, W]  (4 noisy + 1 clean)
        timesteps: [5]
        text_embedding: [seq_len, 4096]
        text_seq_len: int
    """

    def __init__(self, dataroot, text_len=512, traj_dir="pre-encode/ode-trajectories",
                 **kwargs):
        super().__init__()
        traj_path = osp.join(dataroot, traj_dir)
        self.text_len = text_len
        self.file_list = sorted([
            osp.join(traj_path, f) for f in os.listdir(traj_path)
            if f.endswith(".pt")
        ])
        print(f"[TrajectoryDataset] Found {len(self.file_list)} trajectories")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_item = torch.load(self.file_list[index], map_location="cpu",
                               weights_only=True)
        trajectory = data_item["trajectory"].float()  # [5, C, T, H, W]
        timesteps = data_item["timesteps"].float()    # [5]
        text_emb = data_item["text_embedding"].float()  # [seq_len, 4096]

        # Pad text to fixed length
        seq_len = text_emb.shape[0]
        if seq_len < self.text_len:
            pad = torch.zeros(self.text_len - seq_len, text_emb.shape[1])
            text_emb = torch.cat([text_emb, pad], dim=0)
        else:
            text_emb = text_emb[:self.text_len]

        return {
            "trajectory": trajectory,
            "timesteps": timesteps,
            "text_embedding": text_emb,
            "text_seq_len": min(seq_len, self.text_len),
        }


def create_dataloader(opt, cfg, device_num, eval_load_size=None):
    DATALOADER = {
        'VideoLoader': VideoMultiModalDataset,
        'MultiViewLoader': MultiViewVideoDataset,
        'PreEncoded': PreEncodedDataset,
        'Trajectory': TrajectoryDataset,
    }

    print("Start to setup dataloader")
    loader_cls = cfg['class'] if not opt.eval else cfg.get("eval_class", cfg['class'])
    assert loader_cls in DATALOADER.keys(), f'DataLoader {loader_cls} does not exist. Available: {list(DATALOADER.keys())}'
    loader = DATALOADER[loader_cls]

    params = cfg.get('params', {})

    # Handle load_size_list for random load size per batch
    load_size_list = params.get('load_size_list', None)

    dataset = loader(
        dataroot=opt.dataroot,
        eval_load_size=eval_load_size,
        **params
    )

    # Use custom collate function if load_size_list is provided
    collate_fn = RandomSizeCollateFn(load_size_list) if load_size_list else None

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=cfg.get("shuffle", True) and not opt.eval,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=device_num > 1,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    print("Dataloader setup.")
    return dataloader, len(dataset)


def main():
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description='Test video dataloader')
    parser.add_argument('--dataroot', '-d', type=str, default='test_data',
                        help='Path to test data directory')
    parser.add_argument('--config', type=str, default='configs/training/base.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config_path = args.config

    if not osp.isfile(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")
    dataloader_config = config.get('dataloader', {})
    if not dataloader_config:
        print("No dataloader config found in this config file.")
        return

    print(f"Dataloader config: {dataloader_config}")

    dataroot = args.dataroot
    loader_class = dataloader_config.get('class')
    loader_params = dataloader_config.get('params', {})

    if not loader_class:
        print("Dataloader class is missing in config.")
        return

    dataloader_map = {
        'VideoLoader': VideoMultiModalDataset,
        'MultiViewLoader': MultiViewVideoDataset,
    }

    if loader_class not in dataloader_map:
        print(f"Loader class '{loader_class}' is not supported.")
        return

    print(f"Using loader class: {loader_class}")
    print("Loader parameters:")
    for key, value in loader_params.items():
        print(f"  {key}: {value}")

    dataset = dataloader_map[loader_class](
        dataroot=dataroot,
        **loader_params
    )

    print("\nDataset created successfully!")
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        print("\nTesting data loading...")

        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            elif isinstance(value, str):
                print(f"  {key}: '{value}' (text)")
            elif isinstance(value, tuple):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value)}")

        print("\nDataloader test completed successfully!")
    else:
        print("Dataset is empty - no test data found.")


if __name__ == "__main__":
    main()
