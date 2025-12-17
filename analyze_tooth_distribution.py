#!/usr/bin/env python3
"""
analyze_tooth_distribution.py

Analyzes the distribution of tooth numbers (FDI notation) across the dataset splits.
"""
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def parse_label_file(label_path: Path) -> list[str]:
    """Extract FDI tooth numbers from a label file (raw coordinate lines only)."""
    teeth = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # Raw format: FDI xmin ymin xmax ymax (all integers or FDI + integers)
                fdi = parts[0]
                # Check if remaining parts are integers (raw) vs floats (yolo)
                try:
                    # If all coords are large integers, it's raw format
                    coords = [float(p) for p in parts[1:]]
                    if all(c > 10 or c == int(c) for c in coords):  # heuristic: raw coords are typically > 10
                        teeth.append(fdi)
                except ValueError:
                    pass
    return teeth


def analyze_split(split_dir: Path, split_name: str) -> Counter:
    """Count tooth occurrences in a split."""
    label_dir = split_dir / 'Tooth_Labels'
    if not label_dir.exists():
        print(f"Warning: {label_dir} does not exist")
        return Counter()
    
    tooth_counter = Counter()
    image_counter = Counter()  # Count images per tooth
    
    for label_file in label_dir.glob('*.txt'):
        teeth = parse_label_file(label_file)
        tooth_counter.update(teeth)
        # Count unique teeth per image
        for tooth in set(teeth):
            image_counter[tooth] += 1
    
    print(f"\n{'='*60}")
    print(f"{split_name} - Images per tooth number")
    print(f"{'='*60}")
    print(f"{'Tooth (FDI)':<15} {'Image Count':<15} {'Total Instances':<15}")
    print(f"{'-'*60}")
    
    # Sort by tooth number (numeric if possible)
    sorted_teeth = sorted(image_counter.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    
    for tooth in sorted_teeth:
        print(f"{tooth:<15} {image_counter[tooth]:<15} {tooth_counter[tooth]:<15}")
    
    print(f"{'-'*60}")
    print(f"{'Total unique teeth:':<15} {len(image_counter)}")
    print(f"{'Total images:':<15} {len(list(label_dir.glob('*.txt')))}")
    print(f"{'Total instances:':<15} {sum(tooth_counter.values())}")
    
    return image_counter


def main():
    parser = argparse.ArgumentParser(description='Analyze tooth number distribution in dataset')
    parser.add_argument('--dataset-root', default='.', help='Path to dataset root')
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    all_counts = defaultdict(lambda: {'Training': 0, 'Testing': 0, 'Validation': 0})
    
    for split_name in ['Training', 'Testing', 'Validation']:
        split_dir = dataset_root / split_name
        if split_dir.exists():
            counter = analyze_split(split_dir, split_name)
            for tooth, count in counter.items():
                all_counts[tooth][split_name] = count
    
    # Summary across all splits
    print(f"\n{'='*80}")
    print(f"SUMMARY - Images per tooth across all splits")
    print(f"{'='*80}")
    print(f"{'Tooth (FDI)':<15} {'Training':<15} {'Testing':<15} {'Validation':<15} {'Total':<15}")
    print(f"{'-'*80}")
    
    sorted_teeth = sorted(all_counts.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    
    for tooth in sorted_teeth:
        counts = all_counts[tooth]
        total = counts['Training'] + counts['Testing'] + counts['Validation']
        print(f"{tooth:<15} {counts['Training']:<15} {counts['Testing']:<15} {counts['Validation']:<15} {total:<15}")
    
    print(f"{'-'*80}")
    total_train = sum(c['Training'] for c in all_counts.values())
    total_test = sum(c['Testing'] for c in all_counts.values())
    total_val = sum(c['Validation'] for c in all_counts.values())
    print(f"{'TOTAL':<15} {total_train:<15} {total_test:<15} {total_val:<15} {total_train + total_test + total_val:<15}")


if __name__ == '__main__':
    main()
