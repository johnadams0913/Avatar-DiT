#!/usr/bin/env python

import os
import sys
import argparse

# Add engines to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engines import LMDBEngine


def read_lmdb_sample(lmdb_path, key=None, show_first=5):
    """Read and display sample data from LMDB
    
    Args:
        lmdb_path: Path to LMDB database
        key: Specific key to read (if None, read first available)
        show_first: Number of keys to show if key is None
    """
    # Open LMDB
    lmdb_engine = LMDBEngine(lmdb_path, write=False)
    all_keys = lmdb_engine.keys()
    
    print(f"LMDB path: {lmdb_path}")
    print(f"Total entries: {len(all_keys)}")
    print()
    
    if key is None:
        # Show first few keys
        print(f"First {min(show_first, len(all_keys))} keys:")
        for i, k in enumerate(all_keys[:show_first]):
            print(f"  {i+1}. {k}")
        print()
        
        # Use first key as sample
        if all_keys:
            key = all_keys[0]
        else:
            print("No data found in LMDB")
            lmdb_engine.close()
            return
    
    # Read and display data
    print(f"Reading data for key: {key}")
    print("-" * 50)
    
    try:
        data = lmdb_engine[key]
        
        if isinstance(data, dict):
            print("Data format: Dictionary")
            print(f"Fields: {list(data.keys())}")
            print()
            
            for field_name, field_data in data.items():
                print(f"{field_name}:")
                if hasattr(field_data, 'shape'):
                    print(f"  Shape: {field_data.shape}")
                    print(f"  Dtype: {field_data.dtype}")
                    print(f"  Min: {field_data.min():.6f}")
                    print(f"  Max: {field_data.max():.6f}")
                    print(f"  Mean: {field_data.mean():.6f}")
                    
                    # Show a few values
                    if field_data.ndim == 1:
                        print(f"  First 5 values: {field_data[:5]}")
                    elif field_data.ndim == 2:
                        print(f"  First row: {field_data[0][:5] if field_data.shape[1] >= 5 else field_data[0]}")
                    elif field_data.ndim == 3:
                        print(f"  Shape per frame: {field_data[0].shape}")
                        print(f"  First frame first row: {field_data[0][0][:5] if field_data[0].shape[0] >= 5 else field_data[0][0]}")
                else:
                    print(f"  Type: {type(field_data)}")
                    print(f"  Value: {field_data}")
                print()
                
        else:
            print(f"Data type: {type(data)}")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {len(value)}")
            
    except KeyError:
        print(f"Key '{key}' not found in LMDB")
    except Exception as e:
        print(f"Error reading data: {e}")
    
    lmdb_engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read sample data from LMDB")
    parser.add_argument('lmdb_path', type=str, help='Path to LMDB database')
    parser.add_argument('--key', type=str, help='Specific key to read (if not provided, shows first entry)')
    parser.add_argument('--show_first', type=int, default=5, help='Number of keys to display')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.lmdb_path):
        print(f"Error: LMDB path does not exist: {args.lmdb_path}")
    else:
        read_lmdb_sample(args.lmdb_path, args.key, args.show_first)
