#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def rename_mp4_files(root_path):
    """
    Recursively find and rename *.mp4.mp4 files to *.mp4
    
    Args:
        root_path (str): Root directory to search recursively
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        return
    
    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory")
        return
    
    # Find all *.mp4.mp4 files recursively
    mp4_mp4_files = list(root_path.rglob("*.mp4.mp4"))
    
    if not mp4_mp4_files:
        print("No *.mp4.mp4 files found")
        return
    
    print(f"Found {len(mp4_mp4_files)} files to rename:")
    
    for file_path in mp4_mp4_files:
        # Generate new filename by removing the extra .mp4 extension
        new_name = file_path.name[:-4]  # Remove the last .mp4
        new_path = file_path.parent / new_name
        
        print(f"  {file_path} -> {new_path}")
        
        # Check if target file already exists
        if new_path.exists():
            print(f"    Warning: Target file '{new_path}' already exists, skipping...")
            continue
        
        try:
            # Rename the file
            file_path.rename(new_path)
            print(f"    Successfully renamed")
        except Exception as e:
            print(f"    Error renaming file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python rename_mp4_files.py <directory_path>")
        print("Example: python rename_mp4_files.py ./face")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    rename_mp4_files(directory_path)

if __name__ == "__main__":
    main()



