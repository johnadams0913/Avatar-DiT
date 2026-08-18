import cv2
import sys
import json
import os
import os.path as osp
import traceback
import PIL.Image as Image
import zipfile
import pandas as pd
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from glob import glob


def get_video_info(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Failed to open {path}")
            return None
        
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Validation
        if frames <= 0 or width <= 0 or height <= 0:
             print(f"Error: Invalid metadata for {path} (frames={frames}, w={width}, h={height})")
             cap.release()
             return None

        # Try reading the first frame to trigger decoder errors if any
        # This helps identify corrupt files that might print FFMPEG errors
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Failed to read first frame from {path} (corrupt?)")
            return None
            
        return int(frames), int(width), int(height)
    except Exception as e:
        print(f"Exception checking {path}: {e}")
        traceback.print_exc()
        return None


def get_flame_info(path):
    try:
        df = pd.read_parquet(path)
        num_frames = len(df)
        num_dims = df.shape[1] if len(df.shape) > 1 else 1
        
        if num_dims != 112:
            print(f"Error: Flame dimension mismatch for {path}: expected 112, got {num_dims}")
            return None
        
        return num_frames, num_dims
    except Exception as e:
        print(f"Error reading flame parquet {path}: {e}")
        traceback.print_exc()
        return None


def process_single_video(
    video,
    json_base_dir,
    fmt,
    video_key,
    control_key,
    caption_key,
    ref_key,
    face_key,
    track_key,
    flame_key,
    camera_key,
    control_required,
    caption_required,
    face_required,
    flame_required,
    camera_required,    
):
    vpath = osp.abspath(video)
    relative_vpath = osp.relpath(vpath, json_base_dir)

    if fmt == "zip":
        try:
            vzip = zipfile.ZipFile(vpath)
            frame_list = sorted(vzip.namelist())[1:]
            frame_number = len(frame_list)
            width, height = Image.open(vzip.open(frame_list[0])).size
        except Exception as e:
            print(f"Error processing zip {vpath}: {e}")
            return None
    else:
        # Use helper to check video integrity and get info
        info = get_video_info(vpath)
        if info is None:
            return None
        frame_number, width, height = info

    control_path = vpath.replace(f"/{video_key}/", f"/{control_key}/")
    face_path = vpath.replace(f"/{video_key}/", f"/{face_key}/")
    caption_path = vpath.replace(f"/{video_key}/", f"/{caption_key}/").replace(f".{fmt}", ".txt")
    flame_path = vpath.replace(f"/{video_key}/", f"/{flame_key}/").replace(f".{fmt}", ".parquet")
    video_stem = osp.splitext(osp.basename(vpath))[0]
    video_id, camera_id = (video_stem.rsplit("_", 1) + [None])[:2]
    camera_dir = vpath.replace(f"/{video_key}/", f"/{camera_key}/")
    camera_dir = osp.dirname(camera_dir)
    camera_path = osp.join(camera_dir, f"{video_id}.json")
    if control_required and not osp.exists(control_path):
        return None
    if caption_required and not osp.exists(caption_path):
        return None
    if face_required and not osp.exists(face_path):
        return None
    if flame_required and not osp.exists(flame_path):
        return None
    if camera_required:
        if not osp.exists(camera_path) or camera_id is None:
            return None
        with open(camera_path, "r", encoding="utf-8") as f:
            camera_data = json.load(f)
        if str(camera_id) not in camera_data:
            return None

    frame_mismatch = False
    if control_required and osp.exists(control_path):
        # Use helper for control video
        info = get_video_info(control_path)
        if info is None:
            print(f"Warning: Corrupt control video for {relative_vpath}")
            return None
        control_frame_number, _, _ = info
        
        if control_frame_number != int(frame_number):
            print(f"Warning: Frame count mismatch for {relative_vpath}")
            print(f"  Video frames: {int(frame_number)}, Control frames: {control_frame_number}")
            frame_mismatch = True
    
    if face_required and osp.exists(face_path):
        # Use helper for face video
        info = get_video_info(face_path)
        if info is None:
            print(f"Warning: Corrupt face video for {relative_vpath}")
            return None
        face_frame_number, _, _ = info
        
        if face_frame_number != int(frame_number):
            print(f"Warning: Frame count mismatch for {relative_vpath}")
            print(f"  Video frames: {int(frame_number)}, Face frames: {face_frame_number}")
            frame_mismatch = True
    
    if flame_required and osp.exists(flame_path):
        flame_info = get_flame_info(flame_path)
        if flame_info is None:
            print(f"Warning: Corrupt or invalid flame parquet for {relative_vpath}")
            return None
        
        flame_frame_count, flame_dim = flame_info
        if flame_frame_count != int(frame_number):
            print(f"Warning: Frame count mismatch for {relative_vpath}")
            print(f"  Video frames: {int(frame_number)}, Flame frames: {flame_frame_count}")
            frame_mismatch = True
    
    if frame_mismatch:
        return None
    
    if osp.exists(caption_path):
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
    else:
        caption = ""
    
    reference_path = vpath.replace(f"/{video_key}/", f"/{ref_key}/").replace(f".{fmt}", ".png")
    reference_path = osp.relpath(reference_path, json_base_dir) if osp.exists(reference_path) else None

    # print(f"video {relative_vpath} added to list")
    vid_dict = {
        "path": relative_vpath,
        "height": int(height),
        "width": int(width),
        "reference": reference_path,
        "tracking": osp.exists(
            vpath.replace(f"/{video_key}/", f"/{track_key}/").replace(f".{fmt}", ".json")
        ),
        "caption": caption,
        "frame_number": int(frame_number)
    }
    return vid_dict


def process_single_video_worker(args):
    return process_single_video(*args)


def make_video_list(save_path, paths, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    json_base_dir = osp.dirname(osp.abspath(save_path))
    fmt = "mp4"
    video_key = "video"
    control_key = "mesh"
    caption_key = "caption"
    ref_key = "reference"
    face_key = "face-smpl"
    track_key = "track_boxes"
    flame_key = "flame"
    camera_key = "camera"
    
    control_required = True
    caption_required = False
    face_required = False
    flame_required = False
    camera_required = False
    final_video_dict_list = []
    path_counts = {}
    
    # Process each path separately
    with Pool(num_workers) as pool:
        for idx, path in enumerate(paths):
            abs_path = osp.abspath(path)
            print(f"Scanning path: {abs_path}")
            videos = glob(osp.join(abs_path, video_key, f"*.{fmt}"), recursive=True)
            print(f"Found {len(videos)} videos in {abs_path}, processing with {num_workers} workers...")
            
            args_list = [
                (video, json_base_dir, fmt, video_key, control_key, caption_key, ref_key, 
                 face_key, track_key, flame_key, camera_key, control_required, caption_required, 
                 face_required, flame_required, camera_required)
                for video in videos
            ]
            
            results = list(tqdm(pool.imap(process_single_video_worker, args_list), total=len(args_list)))
            path_video_dict_list = [result for result in results if result is not None]
            
            # Save partial result
            part_save_path = f"{save_path}_part{idx}.json"
            print(f"Saving partial result to {part_save_path} with {len(path_video_dict_list)} videos")
            with open(part_save_path, "w") as file:
                json.dump(path_video_dict_list, file, indent=1)
            
            final_video_dict_list.extend(path_video_dict_list)
            path_counts[path] = len(path_video_dict_list)

    print(f"Total valid videos: {len(final_video_dict_list)}")
    print("Video counts per path:")
    for path, count in path_counts.items():
        print(f"  {path}: {count}")
    
    with open(f"{save_path}.json", "w") as file:
        json.dump(final_video_dict_list, file, indent=1)
    
    # Optional: cleanup partial files
    for idx in range(len(paths)):
        part_save_path = f"{save_path}_part{idx}.json"
        if osp.exists(part_save_path):
            os.remove(part_save_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python mk_vid_list.py <save_path> <path1> [path2] ... [--workers N]")
        sys.exit(1)
    
    args = sys.argv[1:]
    num_workers = None
    
    if '--workers' in args:
        workers_idx = args.index('--workers')
        num_workers = int(args[workers_idx + 1])
        args = args[:workers_idx] + args[workers_idx + 2:]
    
    save_path = args[0]
    paths = args[1:]
    
    make_video_list(save_path, paths, num_workers)
