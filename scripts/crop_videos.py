"""
Crop videos to a target resolution with center crop.
Usage:
    python scripts/crop_videos.py --input_dir mio-data --pattern "sway_*.mp4" --output_dir mio-data/sway_cropped --size 544 700
"""
import argparse
import os
import os.path as osp
import subprocess
from glob import glob


def get_video_size(path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0', path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    w, h = map(int, out.split(','))
    return w, h


def crop_video(input_path, output_path, target_w, target_h):
    w, h = get_video_size(input_path)

    # Scale so that the smaller dimension matches target, then center crop
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # Source is wider -> scale height to target_h, crop width
        scale_h = target_h
        scale_w = int(round(w * target_h / h))
    else:
        # Source is taller -> scale width to target_w, crop height
        scale_w = target_w
        scale_h = int(round(h * target_w / w))

    # Ensure even dimensions for ffmpeg
    scale_w = scale_w + (scale_w % 2)
    scale_h = scale_h + (scale_h % 2)

    crop_x = (scale_w - target_w) // 2
    crop_y = (scale_h - target_h) // 2

    vf = f"scale={scale_w}:{scale_h},crop={target_w}:{target_h}:{crop_x}:{crop_y}"

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', vf,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-an', output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--pattern', type=str, default='sway_*.mp4')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--size', type=int, nargs=2, default=[544, 700],
                        help='Target width and height')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos = sorted(glob(osp.join(args.input_dir, args.pattern)))
    print(f"Found {len(videos)} videos matching '{args.pattern}' in {args.input_dir}")

    target_w, target_h = args.size
    for i, vpath in enumerate(videos):
        fname = osp.basename(vpath)
        out_path = osp.join(args.output_dir, fname)
        if osp.exists(out_path):
            continue
        print(f"[{i+1}/{len(videos)}] Cropping {fname} -> {target_w}x{target_h}")
        try:
            crop_video(vpath, out_path, target_w, target_h)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"Done. Cropped videos saved to {args.output_dir}")


if __name__ == '__main__':
    main()
