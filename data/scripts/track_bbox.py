import json
import os
import os.path as osp
import argparse
import cv2

from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv']


def predict_from_video(model, video_path):
    frame_boxes = {}
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mx, my = 0.0, 0.0
    valid_count = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=osp.basename(video_path), leave=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model.track(frame, persist=True, classes=0, verbose=False)
            x1, y1, x2, y2 = results[0].boxes.xyxy.cpu()[0].int().tolist()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            frame_boxes[frame_idx] = {
                "center": [cx, cy],
                "boxes": [x1, y1, x2, y2]
            }
            mx += cx
            my += cy
            valid_count += 1

        except Exception as e:
            print(f"Failed to track bbox at frame {frame_idx} in {video_path}: {e}")

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    model.predictor.trackers[0].reset()

    mean_center = None
    if valid_count > 0:
        mean_center = [int(mx / valid_count), int(my / valid_count)]

    return {"frame_boxes": frame_boxes, "mean_center": mean_center}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track human bounding boxes from videos using YOLO')
    parser.add_argument('--dataroot', '-d', type=str, required=True,
                        help='Path to directory containing video files (e.g. /data/video)')
    parser.add_argument('--save_path', '-s', type=str, default=None,
                        help='Path to save bbox JSON files (default: <dataroot>/../track_boxes)')
    parser.add_argument('--model', '-m', type=str, default='yolo11x.pt',
                        help='YOLO model to use (default: yolo11x.pt)')
    args = parser.parse_args()

    save_path = args.save_path or osp.join(osp.dirname(args.dataroot.rstrip("/")), "track_boxes")

    model = YOLO(args.model)
    os.makedirs(save_path, exist_ok=True)

    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob(osp.join(args.dataroot, ext)))
        videos.extend(glob(osp.join(args.dataroot, ext.upper())))
    videos = sorted(set(videos))

    # Skip already processed videos
    existing = {osp.splitext(f)[0] for f in os.listdir(save_path) if f.endswith(".json")}
    videos = [v for v in videos if osp.splitext(osp.basename(v))[0] not in existing]

    print(f"Found {len(videos)} videos to process (skipped {len(existing)} existing), saving to {save_path}")

    for video_path in tqdm(videos, desc="Processing videos"):
        vid = osp.splitext(osp.basename(video_path))[0]

        result = predict_from_video(model, video_path)
        with open(osp.join(save_path, f"{vid}.json"), "w") as f:
            json.dump(result, f, indent=1)
