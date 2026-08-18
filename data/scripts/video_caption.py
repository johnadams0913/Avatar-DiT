"""
    This script captions videos using QwenVL 3.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_DATA_DIR = "./datasets/seamless_inter"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_MAX_PIXELS = 360 * 420
DEFAULT_FPS = 1.0
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_PROMPT = (
    "Your response should be less than 100 words. "
    "Give a detailed caption for the movement of the person in the video, "
    "including the person's facial expressions and body language."
)

def generate_video_caption(
    video_path,
    model,
    processor,
    prompt,
    max_pixels,
    fps,
    max_new_tokens,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{video_path}",
                    "max_pixels": max_pixels,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)
    inputs.pop("token_type_ids", None)

    # Inference
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""


def save_caption(video_path, caption, caption_dir):
    """Save the video caption to a text file."""
    # Use the fixed caption directory path
    os.makedirs(caption_dir, exist_ok=True)

    # Get just the video ID from the path (e.g., '0001' from '0001.mp4')
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    caption_path = os.path.join(caption_dir, f"{video_id}.txt")

    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(caption)
    return caption_path


def process_video(
    video_path,
    caption_dir,
    model,
    processor,
    prompt,
    max_pixels,
    fps,
    max_new_tokens,
    skip_existing,
):
    """Process a video and generate a single caption."""
    caption_file = os.path.join(
        caption_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.txt"
    )
    if skip_existing and os.path.exists(caption_file):
        print(f"Skipping existing caption: {caption_file}")
        return

    # Generate single caption for the entire video
    caption = generate_video_caption(
        video_path,
        model,
        processor,
        prompt,
        max_pixels,
        fps,
        max_new_tokens,
    )

    # Save the caption
    caption_path = save_caption(video_path, caption, caption_dir)
    # print(f"Caption saved to {caption_path}")
    # print("\nGenerated Caption:")
    # print("-" * 50)
    # print(caption)
    # print("-" * 50)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate video captions using QwenVL 3")
    parser.add_argument('--batch_id', '-bi', type=int, required=True, help='Current batch ID (0-based)')
    parser.add_argument('--batch_num', '-bn', type=int, required=True, help='Total number of batches')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Dataset root directory')
    parser.add_argument('--video-dir', type=str, default=None, help='Video directory (overrides data-dir/video)')
    parser.add_argument('--caption-dir', type=str, default=None, help='Caption output directory (overrides data-dir/caption)')
    parser.add_argument('--model-id', type=str, default=DEFAULT_MODEL_ID, help='QwenVL model id')
    parser.add_argument('--processor-id', type=str, default=None, help='Processor id (defaults to model-id)')
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS, help='Max pixels per frame')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help='Sampling fps')
    parser.add_argument('--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS, help='Max new tokens')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Caption prompt')
    parser.add_argument('--skip-existing', action='store_true', help='Skip videos with existing captions')
    args = parser.parse_args()

    current_batch_id = args.batch_id
    batch_num = args.batch_num

    # Validate arguments
    if batch_num <= 0:
        print("Error: batch_num must be greater than 0")
        sys.exit(1)
    if current_batch_id < 0 or current_batch_id >= batch_num:
        print(f"Error: batch_id ({current_batch_id}) must be between 0 and {batch_num - 1}")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    base_dir = Path(args.video_dir) if args.video_dir else data_dir / "video"
    caption_dir = Path(args.caption_dir) if args.caption_dir else data_dir / "caption"

    # Get all video files recursively
    video_paths = [
        str(path) for path in sorted(base_dir.rglob("*.mp4")) if path.is_file()
    ]
    data_size = len(video_paths)

    if data_size == 0:
        print("No video files found")
        return

    batch_size = max(1, math.ceil(data_size / batch_num))
    start_idx = current_batch_id * batch_size
    end_idx = min(start_idx + batch_size, data_size)
    video_paths = video_paths[start_idx:end_idx]
    data_size = len(video_paths)
    
    if not video_paths:
        print(f"No videos in batch {current_batch_id}")
        return

    print(f"Processing batch {current_batch_id}/{batch_num - 1}")
    print(f"Total videos: {data_size}, batch size: {batch_size}")
    print(f"Processing videos {start_idx} to {end_idx - 1} ({len(video_paths)} videos)")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor_id = args.processor_id or args.model_id
    processor = AutoProcessor.from_pretrained(processor_id)
    model.eval()

    for i, video_path in enumerate(video_paths):
        print(f"\nProcessing video {i}/{data_size}: {video_path}")
        process_video(
            video_path,
            str(caption_dir),
            model,
            processor,
            args.prompt,
            args.max_pixels,
            args.fps,
            args.max_new_tokens,
            args.skip_existing,
        )


if __name__ == "__main__":
    main()