"""
Training script for tooth numbering detection
Full training with GPU
"""

from ultralytics import YOLO

print("=" * 70)
print("TOOTH NUMBERING - FULL TRAINING")
print("=" * 70)
print("\nConfiguration:")
print("  - Device: GPU (automatic detection)")
print("  - Epochs: 100")
print("  - Image size: 640")
print("  - Batch size: 16")
print("  - Classes: 32 (FDI tooth numbering)")
print("\nNote: Focal loss (fl_gamma) not available in this YOLO version")
print("For class imbalance, using data augmentation")
print("=" * 70)

# Load YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train the model with full configuration
results = model.train(
    data='data.yaml',
    epochs=100,        # Full training
    imgsz=640,         # Standard YOLO image size
    batch=16,          # Standard batch size for GPU
    device=0,          # Use GPU (0 = first GPU, 'cpu' for CPU)
    workers=8,         # More workers for GPU
    verbose=True,      # Show detailed output
    patience=50,       # Early stopping patience
    save=True,         # Save checkpoints
    plots=True,        # Generate plots
)

print("\n" + "=" * 70)
print("✓ Training complete!")
print("=" * 70)
print("\nResults saved in: runs/detect/train")
print("\nKey files:")
print("  - Best weights: runs/detect/train/weights/best.pt")
print("  - Last weights: runs/detect/train/weights/last.pt")
print("  - Training curves: runs/detect/train/results.png")
print("  - Confusion matrix: runs/detect/train/confusion_matrix.png")
print("=" * 70)
