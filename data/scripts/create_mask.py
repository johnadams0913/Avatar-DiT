import os
import zipfile
import numpy as np
import os.path as osp
import PIL.Image as Image
import argparse
import cv2

from tqdm import tqdm
from glob import glob
from transformers import AutoModelForImageSegmentation

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

bic = transforms.InterpolationMode.BICUBIC
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

def save_masks(save_path, masks, original_sizes, basenames, camera_dir, use_zip):
    if not isinstance(masks, torch.Tensor):
        masks = torch.stack(masks)
    masks = (masks.clamp(0, 1) * 255.).squeeze(1).cpu().numpy().astype(np.uint8)

    for mask, size, bn in zip(masks, original_sizes, basenames):
        if use_zip:
            # For zip mode, maintain internal directory structure
            subdir = osp.dirname(bn)
            filename = osp.basename(bn)
            output_path = osp.join(save_path, subdir)
            save_file = osp.join(output_path, filename)
        else:
            # Extract path components
            # e.g., /data/.../10001/images_lr/camera_name/xxxx.jpg
            # camera_dir: /data/.../10001/images_lr/camera_name
            # parent_dir: /data/.../10001
            # camera_name: camera_name
            images_lr_dir = osp.dirname(camera_dir)
            parent_dir = osp.dirname(images_lr_dir)
            parent_name = osp.basename(parent_dir)
            camera_name = osp.basename(camera_dir)
            
            # Get path relative to camera directory
            rel_path = osp.relpath(bn, camera_dir)
            
            # Construct output path: save_path/parent_name/masks_lr/camera_name/relative_path
            output_path = osp.join(save_path, parent_name, 'masks_lr', camera_name)
            save_file = osp.join(output_path, rel_path)
            output_path = osp.dirname(save_file)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        mask = Image.fromarray(mask, mode='L').resize((int(size[1]), int(size[0])), Image.Resampling.BICUBIC)
        mask.save(save_file)



class ImageMaskDataset(data.Dataset):
    def __init__(self, dataroot, image_size=1024, use_zip=False):
        super().__init__()
        self.dataroot = osp.abspath(dataroot)
        self.image_size = image_size
        self.use_zip = use_zip
        self.zip_reader = None
        
        if use_zip:
            self.zip_path = self.dataroot
            self.image_files = sorted([
                file for zip in glob(osp.join(self.zip_path, "*.zip"))
                for file in zipfile.ZipFile(zip).namelist() if
                (not file.endswith("/") and file.split(".")[-1] in IMAGE_EXTENSIONS)
            ])
        else:
            self.image_files = []
            for ext in IMAGE_EXTENSIONS:
                self.image_files.extend(glob(osp.join(self.dataroot, f'*.{ext}')))
                self.image_files.extend(glob(osp.join(self.dataroot, f'*.{ext.upper()}')))
            
            for ext in IMAGE_EXTENSIONS:
                self.image_files.extend(glob(osp.join(self.dataroot, '**', f'*.{ext}'), recursive=True))
                self.image_files.extend(glob(osp.join(self.dataroot, '**', f'*.{ext.upper()}'), recursive=True))
            
            self.image_files = sorted(list(set(self.image_files)))
        
        self.data_size = len(self.image_files)
        print(f"Found {self.data_size} images in {'zip files' if use_zip else 'directory'}: {self.dataroot}")

    def fresh_zip_files(self, zipid):
        zip_path = osp.join(self.zip_path, f"{zipid}.zip")
        if not osp.exists(zip_path):
            possible_zips = glob(osp.join(self.zip_path, f"*{zipid}*.zip"))
            if possible_zips:
                zip_path = possible_zips[0]
            else:
                raise FileNotFoundError(f"No zip file found for zipid: {zipid}")
        self.zip_reader = zipfile.ZipFile(zip_path, "r")

    def get_images(self, index):
        filename = self.image_files[index]
        
        if self.use_zip:
            zipid = osp.dirname(filename)
            if not zipid:
                zipid = osp.splitext(osp.basename(filename))[0]
            
            self.fresh_zip_files(zipid)
            img = Image.open(self.zip_reader.open(filename, "r")).convert("RGB")
        else:
            img = Image.open(filename).convert("RGB")
        
        width, height = img.size
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        x = transforms.ToTensor()(img)
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)

        return x, (height, width)


    def __getitem__(self, index):
        while True:
            try:
                img, original_size = self.get_images(index)

                return {
                    "image": img,
                    "size": torch.Tensor(original_size),
                    "filename": self.image_files[index]
                }
            except Exception as e:
                print(f"Cannot handle file {self.image_files[index]} for mask generation due to {e}")
                # Move to next index or wrap around
                index = (index + 1) % len(self.image_files)
                if index == 0:  # If we've cycled through all files
                    raise RuntimeError("No valid images found in dataset")

    def __len__(self):
        return len(self.image_files)



def find_images_lr_dirs(base_path):
    """Find all images_lr subdirectories recursively"""
    images_lr_dirs = []
    for root, dirs, files in os.walk(base_path):
        if 'images_lr' in dirs:
            images_lr_dirs.append(osp.join(root, 'images_lr'))
    return images_lr_dirs


def create_mp4_from_camera(camera_dir, save_path, parent_name, fps=30, keep_images=False):
    """Create MP4 video from mask images for a single camera"""
    camera_name = osp.basename(camera_dir)
    
    # Find all mask images in this camera directory
    mask_files = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        mask_files.extend(sorted(glob(osp.join(camera_dir, f'*.{ext}'))))
    mask_files = sorted(list(set(mask_files)))
    
    if len(mask_files) == 0:
        print(f"No mask files found in {camera_dir}, skipping video creation")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f"Failed to read image {mask_files[0]}, skipping video creation")
        return
    height, width = first_img.shape
    
    # Create video writer with naming format: {parent_name}_{camera_name}.mp4
    # Save directly to save_path
    video_filename = f'{parent_name}_{camera_name}.mp4'
    os.makedirs(save_path, exist_ok=True)
    video_path = osp.join(save_path, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)
    
    print(f"Creating video: {video_filename} ({len(mask_files)} frames at {fps} fps)")
    
    for mask_file in tqdm(mask_files, desc=f"Encoding {video_filename}"):
        frame = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved: {video_path}")
    
    # Delete mask images if keep_images is False
    if not keep_images:
        print(f"Deleting {len(mask_files)} mask images from {camera_dir}")
        for mask_file in mask_files:
            try:
                os.remove(mask_file)
            except:
                pass
        # Remove empty camera directory
        try:
            os.rmdir(camera_dir)
        except OSError:
            pass


def create_mp4_from_masks(masks_lr_dir, save_path, fps=30, keep_images=False):
    """Create MP4 video from mask images in a directory"""
    # Get parent directory name (e.g., 100001)
    parent_name = osp.basename(osp.dirname(masks_lr_dir))
    
    camera_dirs = [d for d in glob(osp.join(masks_lr_dir, '*')) if osp.isdir(d)]
    
    for camera_dir in camera_dirs:
        create_mp4_from_camera(camera_dir, save_path, parent_name, fps, keep_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create masks for images')
    parser.add_argument('--dataroot', '-d', type=str, 
                        default='./datasets/MVHumanNet_original',
                        help='Path to input data directory (default: ./datasets/MVHumanNet_original)')
    parser.add_argument('--save_path', '-s', type=str, required=True,
                        help='Path to save generated masks')
    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Image size for processing (default: 1024)')
    parser.add_argument('--num_threads', '-nt', type=int, default=8,
                        help='Number of threads for data loading (default: 8)')
    parser.add_argument('--use_zip', action='store_true',
                        help='Enable zip-based reading mode instead of regular file system')
    parser.add_argument('--create_mp4', action='store_true',
                        help='Create MP4 videos from masks after processing')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for MP4 videos (default: 30)')
    parser.add_argument('--keep_images', action='store_true',
                        help='Keep mask images after creating MP4 videos (default: delete images)')
    
    args = parser.parse_args()

    model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True,
    ).cuda()

    # Convert paths to absolute for consistent path replacement
    args.dataroot = osp.abspath(args.dataroot)
    args.save_path = osp.abspath(args.save_path)
    
    # Find all images_lr directories
    images_lr_dirs = find_images_lr_dirs(args.dataroot)
    print(f"Found {len(images_lr_dirs)} images_lr directories")
    
    for images_lr_dir in images_lr_dirs:
        print(f"\nProcessing: {images_lr_dir}")
        parent_dir = osp.dirname(images_lr_dir)
        parent_name = osp.basename(parent_dir)
        masks_lr_dir = osp.join(args.save_path, parent_name, 'masks_lr')
        
        # Find all camera directories
        camera_dirs = [d for d in glob(osp.join(images_lr_dir, '*')) if osp.isdir(d)]
        
        if len(camera_dirs) == 0:
            print(f"No camera directories found in {images_lr_dir}, skipping...")
            continue
        
        print(f"Found {len(camera_dirs)} camera directories")
        
        # Process each camera separately
        for camera_dir in camera_dirs:
            camera_name = osp.basename(camera_dir)
            print(f"\n  Processing camera: {camera_name}")
            
            # Create dataset for this specific camera
            dataset = ImageMaskDataset(camera_dir, args.image_size, args.use_zip)
            if len(dataset) == 0:
                print(f"  No images found in {camera_dir}, skipping...")
                continue
            
            dataloader = data.DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_threads,
                drop_last=False,
                pin_memory=True,
                prefetch_factor=2 if args.num_threads > 0 else None,
            )

            with torch.no_grad():
                for idx, batch_data in enumerate(tqdm(dataloader, desc=f"  {parent_name}/{camera_name}")):
                    img = batch_data["image"].cuda()
                    original_size = batch_data["size"]
                    bn = batch_data["filename"]

                    predicted = model(img)[-1].sigmoid()
                    
                    original_size = original_size.numpy().astype(np.int32)
                    save_masks(args.save_path, predicted, original_size, bn, 
                              camera_dir, args.use_zip)
            
            # Create MP4 video for this camera immediately after processing
            if args.create_mp4:
                output_camera_dir = osp.join(masks_lr_dir, camera_name)
                if osp.exists(output_camera_dir):
                    print(f"  Creating MP4 for camera: {camera_name}")
                    create_mp4_from_camera(output_camera_dir, args.save_path, parent_name, args.fps, args.keep_images)
                else:
                    print(f"  Warning: camera directory not found: {output_camera_dir}")