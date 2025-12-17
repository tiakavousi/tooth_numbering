# convert_to_tooth_labels

Utility to convert Key Points Annotations JSON bounding boxes into per-image tooth-number label files.

Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the script (dataset root defaults to current directory):

```bash
python convert_to_tooth_labels.py --dataset-root "/Users/tayebekavousi/Desktop/Dataset" --mode both
```

Options:
- `--mode`: `raw`, `yolo`, or `both` (default `both`).
- `--label-dir-name`: directory created inside each split (default `Tooth_Labels`).
- `--remap-classes`: map FDI numbers to 0-based indices and write `classes.txt`.
