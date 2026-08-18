import numpy as np
import os
import os.path as osp
import PIL.Image as Image
import imageio
import json
import cv2
import subprocess
import pandas as pd

from glob import glob
from datetime import datetime
from data.utils import track_crop
from .imaginary_caption import create_model

import torch
import torchvision.transforms as transforms



maxium_resolution = 4096
token_length = int(256 ** 0.5)
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

vl_archs = {
    "qwen-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}


def exists(v):
    return v is not None

def path_exists(path):
    """Check if path exists (both not None and file exists)"""
    return exists(path) and osp.exists(path)

def is_video_file(path):
    """Check if path is a video file"""
    return osp.isfile(path) and path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

def get_video_id(path):
    """Extract video ID from file path"""
    return osp.splitext(osp.basename(path))[0]

def denormalize(x: torch.Tensor):
    return ((x.float() + 1.) * 127.5).clamp(0, 255)

def resize_image(img, new_size):
    w, h = img.size
    if w > h:
        img = transforms.Resize((int(h / w * new_size), new_size))(img)
    else:
        img = transforms.Resize((new_size, int(w / h * new_size)))(img)
    return img

def pad_image_to_square(image: Image, original_shape=False):
    width, height = image.size
    max_dim = max(width, height)
    padding = (0, 0, 0)
    square_image = Image.new('RGB', (max_dim, max_dim), padding)
    left = (max_dim - width) / 2
    top = (max_dim - height) / 2
    square_image.paste(image, (int(left), int(top)))

    if original_shape:
        return square_image, (int(left), int(top), width, height)
    else:
        return square_image

def resize(image, final_size=(512, 768), is_ref=False):
    """
    Resize image to target size with center crop.
    Match the behavior of dataloader's resize_with_crop function.
    """
    w, h = image.size
    target_w, target_h = final_size
    
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
        image = image.resize((target_w, scaled_h), Image.LANCZOS)
        image = image.crop((0, start_y, target_w, end_y))
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
        image = image.resize((scaled_w, target_h), Image.LANCZOS)
        image = image.crop((start_x, 0, end_x, target_h))
    
    # Final resize to ensure exact dimensions
    return image.resize(final_size, Image.LANCZOS)

def crop_image_from_square(square_image, original_dim):
    left, top, width, height = original_dim
    return square_image.crop((left, top, left + width, top + height))



def to_tensor(x, inverse=False, normalize=False):
    x = transforms.ToTensor()(x).unsqueeze(0)
    if normalize:
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
    return x if not inverse else 1-x

def to_numpy(x, denormalize=False):
    if denormalize:
        x = (x.clamp(-1, 1) + 1.) * 127.5
    else:
        x = x.clamp(0, 1) * 255.
    return x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def image_preprocessing(control, reference, resolution):
    control = to_tensor(resize(control, final_size=resolution))
    reference = to_tensor(resize(reference, final_size=resolution), normalize=True)
    return control, reference


def save_video(frames, filename, fps=24):
    writer = imageio.get_writer(filename, fps=fps, codec='libx264', format='FFMPEG')
    print(f"frames.shape: {frames.shape}")
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def get_max_chunk_num(video_path, frame_length, stride, start_frame=0):
    """Return the maximum chunk_num the input video can support."""
    if not path_exists(video_path) or not is_video_file(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    available = max(total - start_frame, 0)
    if stride <= 0 or available < frame_length:
        return 1 if available >= frame_length else 0
    return max((available - frame_length) // stride + 1, 1)


def read_video_frames_cv2(video_path, num_frames, start_frame=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    read_count = 0
    while read_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        read_count += 1
    
    cap.release()
    return frames

def read_flame_sequence(flame_path, video_length, start_frame=0):
    if not path_exists(flame_path):
        return torch.zeros(video_length, 112).float()
    
    if flame_path.endswith('.parquet') and osp.isfile(flame_path):
        print(f"load FLAME data from parquet file {flame_path}")
        df = pd.read_parquet(flame_path)
        motion_code = df.values.astype(np.float32)
        return torch.from_numpy(motion_code[start_frame: start_frame + video_length]).float()
    print(f"load FLAME data from {flame_path}")
    data = torch.load(flame_path)
    return data[start_frame: start_frame + video_length].float()

def modify_flame_parameters(flame, enable_gpose=False, enable_left_eye=False,
                            enable_right_eye=False, enable_mouth=False,
                            gpose_control=0.0, left_eye_h=0.0, right_eye_h=0.0, mouth_open=0.0,
                            use_ramp=True):
    """
    Flame parameter structure:
        - 100-103: gpose (global pose, 3D)
        - 103: jaw pose
        - 106-112: eye poses (3D for left eye, 3D for right eye)

    Args:
        use_ramp: If True, gradually transition from 0 to target value.
                  If False, use constant target value for all frames.
    """

    if flame is None:
        return flame

    if not (enable_gpose or enable_left_eye or enable_right_eye or enable_mouth):
        return flame

    modified_flame = flame.clone()
    video_length = modified_flame.shape[0]

    modified_flame[:, :100] = 0.0

    if use_ramp:
        scale = torch.linspace(0, 1, video_length, device=modified_flame.device)
    else:
        scale = torch.ones(video_length, device=modified_flame.device)

    if enable_gpose and gpose_control != 0.0:
        modified_flame[:, 102] = scale * gpose_control

    if enable_left_eye and left_eye_h != 0.0:
        modified_flame[:, 107] = scale * left_eye_h

    if enable_right_eye and right_eye_h != 0.0:
        modified_flame[:, 110] = scale * right_eye_h

    if enable_mouth and mouth_open != 0.0:
        modified_flame[:, 103] = scale * mouth_open

    return modified_flame

def read_video_tensor(video_path, video_length, target_size, start_frame=0, 
                      normalize=False, default_value=None, use_custom_resize=True):
    if not path_exists(video_path):
        if default_value == 'zeros':
            return torch.zeros(video_length, 3, *target_size)
        return None
    
    frames = read_video_frames_cv2(video_path, video_length, start_frame)
    resize_func = (lambda f: resize(f, target_size)) if use_custom_resize else (lambda f: f.resize(target_size, Image.LANCZOS))
    
    return torch.cat([to_tensor(resize_func(frame), normalize=normalize) for frame in frames], 0)

def process_frame_with_crop(frame, resolution, center, normalize=False):
    """Process a frame with optional track crop and convert to tensor"""
    processed = track_crop(frame, center) if exists(center) else resize(frame, resolution)
    return to_tensor(processed, normalize=normalize)

def get_image_files(directory_path, video_length):
    """Get sorted image files from directory"""
    image_files = [file for ext in IMAGE_EXTENSIONS for file in glob(osp.join(directory_path, f"*.{ext}"))]
    image_files = sorted(image_files)
    if len(image_files) < video_length:
        raise ValueError(f"Not enough frames: found {len(image_files)}, need {video_length}")
    return image_files

def read_sequence_from_source(source_path, video_length, resolution, center=None, 
                               normalize=False, start_frame=0):
    if is_video_file(source_path):
        frames = read_video_frames_cv2(source_path, video_length, start_frame)
        tensor_frames = torch.cat([process_frame_with_crop(f, resolution, center, normalize) for f in frames], 0)
        return tensor_frames, None
    else:
        image_files = get_image_files(source_path, video_length)
        tensor_frames = torch.cat([
            process_frame_with_crop(Image.open(image_files[idx]).convert("RGB"), resolution, center, normalize)
            for idx in range(video_length)
        ], 0)
        return tensor_frames, image_files

def read_smpl_params(subject_id=None, video_length=None, start_frame=0, mode="multiview", smpl_path=None):
    if smpl_path and osp.exists(smpl_path):
        smpl_dir = smpl_path
    elif subject_id:
        smpl_dir = osp.join(f"presets/{mode}/smpl-param", subject_id)
    else:
        return None
    
    if not osp.exists(smpl_dir):
        return None
    
    try:
        vec_list = []
        for i in range(video_length):
            frame_idx = start_frame + i
            json_path = osp.join(smpl_dir, f"{frame_idx:06d}.json")
            
            if not osp.exists(json_path):
                print(f"SMPL param file not found: {json_path}")
                return None
            
            with open(json_path, "r") as f:
                data = json.load(f)
            
            item = data[0] if isinstance(data, list) and len(data) > 0 else data
            
            Rh = item.get("Rh", [[0.0, 0.0, 0.0]])[0]
            Th = item.get("Th", [[0.0, 0.0, 0.0]])[0]
            poses = item.get("poses", [[0.0] * 87])[0]
            shapes = item.get("shapes", [[0.0] * 10])[0]
            
            combined = Rh + Th + poses + shapes
            vec_list.append(combined)
        
        return torch.tensor(vec_list, dtype=torch.float32)
    except Exception as e:
        print(f"Failed to load SMPL params from {smpl_dir}: {e}")
        return None

def read_face_mode_data(control_path, video_path, resolution, video_length, flame_path=None, gt_start=0, face_path=None):
    controls = read_video_tensor(control_path, video_length, resolution, gt_start, normalize=False, default_value=None, use_custom_resize=True)
    face = read_video_tensor(face_path, video_length, (512, 512), gt_start, normalize=False, default_value=None, use_custom_resize=False)
    video = read_video_tensor(video_path, video_length, resolution, gt_start, normalize=True, default_value='zeros', use_custom_resize=True)
    flame = read_flame_sequence(flame_path, video_length, gt_start)
    return controls, video, face, flame, video_path

def read_multiview_video_data(control_path, video_path, resolution, video_length, gt_start=0, mode="multiview", smpl_path=None, flame_path=None):
    controls = read_video_tensor(control_path, video_length, resolution, gt_start, normalize=False, default_value=None, use_custom_resize=True)
    video = read_video_tensor(video_path, video_length, resolution, gt_start, normalize=True, default_value='zeros', use_custom_resize=True)
    
    flame = get_flame_for_paths(control_path, video_path, None, video_length, gt_start, flame_path, None) if path_exists(flame_path) else None
    
    smpl_params = None
    if path_exists(smpl_path):
        smpl_params = read_smpl_params(video_length=video_length, start_frame=gt_start, mode=mode, smpl_path=smpl_path)
    else:
        for path in [control_path]:
            if exists(path):
                smpl_params = read_smpl_params(osp.basename(path).split('_')[0], video_length, gt_start, mode)
                break
        if smpl_params is None and exists(video_path):
             smpl_params = read_smpl_params(osp.basename(video_path).split('_')[0], video_length, gt_start, mode)

    return controls, [video, None], smpl_params, flame

def read_image_sequence_tensor(image_files, video_length, resolution, center=None, normalize=True):
    """Read tensor from image file list with optional cropping."""
    return torch.cat([
        process_frame_with_crop(Image.open(image_files[idx]).convert("RGB"), resolution, center, normalize)
        for idx in range(video_length)
    ])

def read_pose_sequence(control_path, resolution, video_length, video_path=None, track=False, gt_start=0):
    try:
        center = json.load(open(video_path.replace("frame", "track_boxes") + ".json"))["mean_center"]
    except:
        center = None

    controls, control_files = read_sequence_from_source(
        control_path, video_length, resolution, center, normalize=False, start_frame=gt_start
    )
    
    if exists(video_path):
        if is_video_file(video_path):
            video, _ = read_sequence_from_source(video_path, video_length, resolution, center, normalize=True, start_frame=gt_start)
        elif osp.isdir(video_path):
            if is_video_file(control_path):
                raise ValueError("Cannot use video file for control_path with directory for video_path")
            image_files = [osp.join(video_path, osp.basename(file)) for file in control_files]

            video = read_image_sequence_tensor(image_files, video_length, resolution, center, normalize=True)
        else:
            raise ValueError(f"video_path must be a valid video file or directory: {video_path}")
    else:
        if is_video_file(control_path):
            frames = read_video_frames_cv2(control_path, video_length, start_frame=gt_start)
            video = torch.cat([to_tensor(resize(frame, resolution), normalize=True) for frame in frames])
        else:
            image_files = [file.replace('smpl', '') for file in control_files]
            video = read_image_sequence_tensor(image_files, video_length, resolution, center, normalize=True)
    
    return controls, [video, center]    

def load_camera_params(control_path, camera_view1, mode="multiview"):
    """Load camera parameters from JSON files"""
    if not exists(camera_view1) or not exists(control_path):
        return None
    
    subject_id = osp.basename(control_path).split('_')[0]
    camera_file = osp.join(f"presets/{mode}/camera", f"{subject_id}.json")
    
    if not osp.exists(camera_file):
        return None
    
    with open(camera_file, 'r') as f:
        camera_data = json.load(f)
    
    if camera_view1 not in camera_data:
        return None
    
    camera_params = []
    for view in [camera_view1]:
        cam_info = camera_data[view]
        scale = cam_info.get("scale", 1.0)

        extrinsic_tensor = torch.tensor(cam_info["extrinsic"], dtype=torch.float32).reshape(4, 4)
        intrinsic_tensor = torch.tensor(cam_info["intrinsic"], dtype=torch.float32).reshape(3, 3)
        distortion = cam_info.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])

        k1, k2, p1, p2, k3 = distortion
        radial = 1.0 + k1 + k2 + k3
        intrinsic_tensor[0, 0] *= radial
        intrinsic_tensor[1, 1] *= radial
        intrinsic_tensor[0, 2] += p1 * intrinsic_tensor[0, 0]
        intrinsic_tensor[1, 2] += p2 * intrinsic_tensor[1, 1]

        intrinsic_tensor[:2] *= scale
        extrinsic_tensor[:3, 3] *= scale

        camera_param = torch.cat([extrinsic_tensor.reshape(-1, 16), intrinsic_tensor.reshape(-1, 9)], dim=-1)
        camera_params.append(camera_param.squeeze(0))

    camera_tensor = torch.stack(camera_params, dim=0).to(dtype=torch.float32)
    return camera_tensor

@torch.no_grad()
def video_preprocessing(
        control_path,
        reference,
        resolution,
        video_length,
        video_path = None,
        face_path = None,
        mode = "demo",
        custom_video = None,
        custom_gt_video = None,
        camera_view1 = None,
        smpl_path = None,
        flame_path = None,
        enable_gpose = False,
        enable_left_eye = False,
        enable_right_eye = False,
        enable_mouth = False,
        gpose_control = 0.0,
        left_eye_h = 0.0,
        right_eye_h = 0.0,
        mouth_open = 0.0,
        flame_ramp = True,
        gt_start = 0,
):
    center = None
    face = None
    flame = None
    source_video_path = None
    camera = None
    smpl_params = None

    if custom_video is not None:
        control_path = custom_video
    
    if custom_gt_video is not None:
        video_path = custom_gt_video
    
    source_video_path = video_path if exists(video_path) and is_video_file(video_path) else None

    if mode in ("face",):
        controls, video, face, flame, source_video_path = read_face_mode_data(
            control_path, video_path, resolution, video_length, gt_start=gt_start, face_path=face_path, flame_path=flame_path
        )
    elif mode in ("multiview", "mv-long"):
        mv_preset = "multiview"
        controls, [video, center], smpl_params, flame = read_multiview_video_data(
            control_path, video_path, resolution, video_length, gt_start, mode=mv_preset, smpl_path=smpl_path, flame_path=flame_path
        )
        camera = load_camera_params(control_path, camera_view1, mv_preset)
    else:
        controls, [video, center] = read_pose_sequence(
            control_path, resolution, video_length, video_path, track=False, gt_start=gt_start
        )

    reference = reference if exists(reference) else None
    if isinstance(reference, (list, tuple)):
        reference = [item for item in reference if exists(item)]

    reference_tensors = []
    if isinstance(reference, (list, tuple)):
        for item in reference:
            if isinstance(item, str):
                if not osp.exists(item):
                    continue
                item = Image.open(item).convert("RGB")
            if exists(center):
                item = track_crop(item, (item.size[0]//2, item.size[1]//2))
            reference_tensors.append(to_tensor(resize(item, resolution), normalize=True))
        reference = torch.cat(reference_tensors, dim=0) if reference_tensors else None
    elif exists(reference):
        if isinstance(reference, str) and osp.exists(reference):
            reference = Image.open(reference).convert("RGB")
        if exists(center):
            reference = track_crop(reference, (reference.size[0]//2, reference.size[1]//2))
        reference = to_tensor(resize(reference, resolution), normalize=True)
    else:
        reference = None

    if not exists(reference):
        reference = torch.zeros([1, 3, resolution[1], resolution[0]]).cuda()
    
    flame = modify_flame_parameters(flame, enable_gpose, enable_left_eye, enable_right_eye, enable_mouth,
                                    gpose_control, left_eye_h, right_eye_h, mouth_open, flame_ramp)
    
    return controls, reference, video, face, flame, source_video_path, camera, smpl_params


def to_cpu_tensor(tensor):
    """Convert tensor to CPU if it's a tensor"""
    return tensor.cpu() if isinstance(tensor, torch.Tensor) and exists(tensor) else tensor

def trim_multiview_tensors(frames, video, poses, reference, face):
    """Trim tensors for multiview mode"""
    num_frames = frames.size(0)
    if exists(video) and video.size(0) >= num_frames * 2:
        video = video[:num_frames]
    if exists(poses) and poses.size(0) >= num_frames * 2:
        poses = poses[:num_frames]
    if exists(reference) and reference.size(0) > 1:
        reference = reference[:1]
    if exists(face) and face.size(0) >= num_frames * 2:
        face = face[:num_frames]
    return video, poses, reference, face

def get_min_frame_count(frames, video, poses, face):
    """Get minimum frame count among all inputs"""
    counts = [frames.size(0)]
    for tensor in [video, poses, face]:
        if exists(tensor):
            counts.append(tensor.size(0))
    return min(counts)

def img2video(frames, poses, reference, video=None, face=None, save_path=None, fps=30, mode="pose", source_video_path=None, app_mode=None, start_frame=0):
    current_time = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    filename = f"{save_path if exists(save_path) else f'video_outputs/{current_time}'}.mp4"

    frames, poses, reference, video, face = [to_cpu_tensor(t) for t in [frames, poses, reference, video, face]]

    if app_mode in ("multiview", "mv-long"):
        video, poses, reference, face = trim_multiview_tensors(frames, video, poses, reference, face)

    min_frames = get_min_frame_count(frames, video, poses, face)
    frames, video, poses, face = [t[:min_frames] if exists(t) else t for t in [frames, video, poses, face]]
        
    concat_list = [frames]
    if exists(video):
        concat_list.insert(0, video)
    if exists(poses):
        poses = (poses[:,:3].clamp(0, 1) - 0.5) * 2 if mode == "pose" else poses
        reference = reference.repeat(poses.size(0), 1, 1, 1)
        concat_list.extend([poses, reference])
    
    if exists(face) and app_mode == "face":
        face = (face.clamp(0, 1) - 0.5) * 2
        target_h, target_w = frames.shape[2:]
        face_h, face_w = face.shape[2:]
        
        if face_h != target_h or face_w != target_w:
            pad_h, pad_w = target_h - face_h, target_w - face_w
            face = torch.nn.functional.pad(
                face, 
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), 
                mode='constant', value=1.0
            )
        
        face = face[:frames.size(0)]
        concat_list.insert(1, face)
    
    concat_tensor = torch.cat(concat_list, dim=3)
    frames = denormalize(concat_tensor).permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
    
    if len(frames.shape) != 4:
        raise ValueError(f"Invalid frames shape after processing: {frames.shape}, expected (N, H, W, 3)")
    if frames.shape[3] != 3:
        raise ValueError(f"Invalid channel count: {frames.shape[3]}, expected 3")
    
    should_add_audio = path_exists(source_video_path) and is_video_file(source_video_path)

    # When merging audio from source, use source fps to keep audio-video sync
    # (frames are extracted 1:1 from source, so playback rate must match source)
    save_fps = fps
    source_fps = None
    if should_add_audio:
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', source_video_path]
            fps_out = subprocess.check_output(probe_cmd).decode().strip()
            if '/' in fps_out:
                num, den = map(int, fps_out.split('/'))
                source_fps = num / den
            else:
                source_fps = float(fps_out)
            if source_fps and source_fps > 0:
                save_fps = source_fps
                print(f"Using source video fps ({source_fps}) for audio sync (user fps: {fps})")
        except Exception as e:
            print(f"Failed to probe source fps: {e}")

    temp_filename = filename.replace(".mp4", "_temp.mp4") if should_add_audio else filename
    save_video(frames, temp_filename, fps=save_fps)

    if should_add_audio and source_fps and source_fps > 0:
        try:
            start_time = start_frame / source_fps

            check_audio_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                              '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', source_video_path]
            result = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=5)

            if result.stdout.strip() == 'audio':
                merge_cmd = ['ffmpeg', '-y', '-i', temp_filename, '-ss', f"{start_time:.3f}", '-i', source_video_path,
                            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', filename]
                subprocess.run(merge_cmd, check=True, capture_output=True, timeout=30)
                os.remove(temp_filename)
                print(f"Audio track added from source video: {source_video_path} (Start time: {start_time:.3f}s)")
            elif temp_filename != filename:
                os.rename(temp_filename, filename)
        except Exception as e:
            print(f"Failed to add audio track: {e}")
            if temp_filename != filename and osp.exists(temp_filename):
                os.rename(temp_filename, filename)
    
    return filename
