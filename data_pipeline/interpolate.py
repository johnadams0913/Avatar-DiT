import os
import numpy as np
import argparse
from tqdm import tqdm
from engines import LMDBEngine


def interpolate_motioncode(motioncode, source_fps, target_fps):
    original_length = motioncode.shape[0]
    target_length = int(original_length / source_fps * target_fps)
    
    original_time = np.linspace(0, 1, original_length)
    target_time = np.linspace(0, 1, target_length)
    
    motioncode_interpolated = np.zeros((target_length, motioncode.shape[1]))
    for i in range(motioncode.shape[1]):
        motioncode_interpolated[:, i] = np.interp(target_time, original_time, motioncode[:, i])
    return motioncode_interpolated


def interpolate_mask(mask, source_fps, target_fps):
    original_length = mask.shape[0]
    target_length = int(original_length / source_fps * target_fps)
    
    original_time = np.linspace(0, 1, original_length)
    target_time = np.linspace(0, 1, target_length)
    
    if mask.ndim == 1:
        mask_interpolated = np.interp(target_time, original_time, mask)
        mask_interpolated = (mask_interpolated > 0.5).astype(mask.dtype)
    else:
        mask_interpolated = np.zeros((target_length, mask.shape[1]))
        for i in range(mask.shape[1]):
            mask_interpolated[:, i] = np.interp(target_time, original_time, mask[:, i])
    return mask_interpolated


def main(source_lmdb, target_lmdb, source_fps, target_fps):
    print(f"Reading from: {source_lmdb}")
    print(f"Writing to: {target_lmdb}")
    print(f"Interpolating from {source_fps} fps to {target_fps} fps")
    
    lmdb_read = LMDBEngine(source_lmdb, write=False)
    lmdb_write = LMDBEngine(target_lmdb, write=True)
    
    all_keys = lmdb_read.keys()
    print(f"Total entries: {len(all_keys)}")
    
    for key in tqdm(all_keys, desc="Interpolating"):
        data = lmdb_read[key]
        
        new_data = {}
        
        if 'motioncode' in data:
            motioncode = data['motioncode'].astype(np.float32)
            motioncode_interp = interpolate_motioncode(motioncode, source_fps, target_fps)
            new_data['motioncode'] = motioncode_interp.astype(np.float16)
        
        if 'mask' in data:
            mask = data['mask']
            mask_interp = interpolate_mask(mask, source_fps, target_fps)
            new_data['mask'] = mask_interp
        
        if 'shapecode' in data:
            new_data['shapecode'] = data['shapecode']
        
        if 'audio' in data:
            new_data['audio'] = data['audio']
        
        lmdb_write.dump(key, new_data)
    
    lmdb_read.close()
    lmdb_write.close()
    
    print("Interpolation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolate motioncode and mask in LMDB')
    parser.add_argument('--source_lmdb', type=str, default='data_lmdb', help='Source LMDB path')
    parser.add_argument('--target_lmdb', type=str, default='data_lmdb_interpolated', help='Target LMDB path')
    parser.add_argument('--source_fps', type=int, default=25, help='Source FPS')
    parser.add_argument('--target_fps', type=int, default=30, help='Target FPS')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_lmdb):
        print(f"Error: Source LMDB path does not exist: {args.source_lmdb}")
    else:
        main(args.source_lmdb, args.target_lmdb, args.source_fps, args.target_fps)