"""
Script to clean YOLO label files by removing FDI tooth numbers with absolute coordinates
and keeping only the class indices with normalized coordinates.
"""

import os
from pathlib import Path


def is_normalized_line(line):
    """ 
    Check if a line contains normalized coordinates (values between 0 and 1).
    
    Args:
        line: String line from label file
    
    Returns:
        bool: True if line has normalized coordinates, False otherwise
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    
    try:
        # Check if the bbox coordinates are normalized (between 0 and 1)
        values = [float(p) for p in parts[1:]]
        return all(0 <= v <= 1 for v in values)
    except ValueError:
        return False


def clean_label_file(file_path, backup=True):
    """
    Clean a single label file by keeping only normalized coordinate lines.
    
    Args:
        file_path: Path to the label file
        backup: If True, create a .bak backup before modifying
    """
    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Filter to keep only normalized lines
    normalized_lines = [line for line in lines if is_normalized_line(line)]
    
    # Write cleaned lines back
    with open(file_path, 'w') as f:
        f.writelines(normalized_lines)
    
    return len(lines), len(normalized_lines)


def clean_all_labels(dataset_path, backup=True):
    """
    Clean all label files in train and val folders.
    
    Args:
        dataset_path: Path to dataset folder containing labels/train and labels/val
        backup: If True, create backups before modifying
    """
    dataset_path = Path(dataset_path)
    
    # Process both train and val folders
    for split in ['train', 'val']:
        labels_dir = dataset_path / 'labels' / split
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping...")
            continue
        
        # Get all .txt files
        txt_files = list(labels_dir.glob('*.txt'))
        print(f"\nProcessing {len(txt_files)} files in {split} set...")
        
        total_original = 0
        total_cleaned = 0
        
        for txt_file in txt_files:
            original_count, cleaned_count = clean_label_file(txt_file, backup)
            total_original += original_count
            total_cleaned += cleaned_count
        
        print(f"  Original lines: {total_original}")
        print(f"  Cleaned lines: {total_cleaned}")
        print(f"  Removed lines: {total_original - total_cleaned}")
    
    print("\n✓ Cleaning complete!")


if __name__ == "__main__":
    # Path to dataset folder
    dataset_path = "dataset"
    
    # Clean all labels (with backup by default)
    clean_all_labels(dataset_path, backup=True)
    
    print("\nTo restore from backup, run:")
    print("  find dataset/labels -name '*.bak' -exec sh -c 'mv \"$1\" \"${1%.bak}\"' _ {} \\;")
