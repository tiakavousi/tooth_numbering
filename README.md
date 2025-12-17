# Tooth Numbering Detection with YOLOv8

Automated tooth detection and numbering system using YOLOv8 object detection. Detects and classifies teeth according to the FDI (Fédération Dentaire Internationale) numbering system (teeth 11-48).

## Dataset

- **Classes**: 32 permanent teeth (FDI notation: 11-48)
- **Training images**: 616
- **Validation images**: 136
- **Format**: YOLO format with normalized bounding boxes

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- See `requirements.txt` for full list

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tooth_numbering.git
cd tooth_numbering
```

### 2. Extract Dataset

The dataset is provided as a compressed zip file. Extract it:

```bash
unzip dataset.zip
```

This will create the `dataset/` folder with the following structure:
```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

### 3. Create Virtual Environment

```bash
python3 -m venv torch-env
source torch-env/bin/activate  # On macOS/Linux
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics torch torchvision opencv-python
```

## Training

### GPU Support

**Apple Silicon (M1/M2/M3 Macs):**
- ✅ GPU acceleration supported via MPS (Metal Performance Shaders)
- Training time: ~3-6 hours for 100 epochs
- No additional setup needed - auto-detected


### Run Training

```bash
python train_test.py
```

**Training Configuration:**
- Model: YOLOv8 nano (yolov8n.pt)
- Epochs: 100
- Image size: 640x640
- Batch size: 16 (GPU) / 4 (CPU)
- Dataset: dataset/data.yaml

### Training Outputs

Results are saved to `runs/detect/train/`:
- `weights/best.pt` - Best model checkpoint (use for inference)
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training curves (loss, mAP, etc.)
- `confusion_matrix.png` - Per-class performance
- `results.csv` - Detailed metrics per epoch

## Class Mapping

The model outputs class IDs 0-31, which map to FDI tooth numbers:

```
0→11, 1→12, 2→13, 3→14, 4→15, 5→16, 6→17, 7→18,
8→21, 9→22, 10→23, 11→24, 12→25, 13→26, 14→27, 15→28,
16→31, 17→32, 18→33, 19→34, 20→35, 21→36, 22→37, 23→38,
24→41, 25→42, 26→43, 27→44, 28→45, 29→46, 30→47, 31→48
```

See `classes.txt` for the full mapping.

## Project Structure

```
tooth_numbering/
├── dataset/                    # Training data (extracted from zip)
│   ├── images/
│   └── labels/
├── data.yaml                   # Dataset configuration
├── train_test.py              # Training script
├── yolov8n.pt                 # Pretrained YOLO model
├── classes.txt                # FDI tooth number mappings
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```