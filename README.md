# Magellon Training — Cryo-EM Particle Picker

Standalone training package for Magellon's ML particle picker. Designed to run on Lightning.ai, RunPod, or any GPU machine without needing the full Rust codebase.

## Quick Start (Lightning.ai)

```bash
# In your Lightning Studio terminal:
git clone https://github.com/khoshbin/magellon-training.git
cd magellon-training
bash setup.sh
```

That's it — `setup.sh` handles everything: installs deps, downloads data, converts to YOLO format, trains YOLOv11-nano (smoke test), then RF-DETR.

## Manual Steps

```bash
pip install -r requirements.txt

# Download CryoPPP EMPIAR-10017 (476 MB)
bash fetch_data.sh

# Convert to YOLO format
python cryoppp_to_coco.py --input data/cryoppp/lite/10017 --output data/training/10017

# Train (pick one):
python train_yolov11.py --data data/training/10017/dataset.yaml --model yolo11n --epochs 10 --device 0
python train_rfdetr.py --data data/training/10017 --model base --epochs 30 --device 0
python train_rtdetr.py --data data/training/10017/dataset.yaml --epochs 30 --device 0

# Export + validate ONNX
python export_onnx.py --model runs/cryoppp-picker/*/weights/best.pt --validate --data data/cryoppp/lite/10017
```

## GPU Requirements

| Model | VRAM | Time (84 images, 10 epochs) |
|-------|------|-----------------------------|
| YOLOv11-nano | 2 GB | ~3 min on T4 |
| YOLOv11-small | 4 GB | ~5 min on T4 |
| RF-DETR-Base | 10-12 GB | ~15 min on T4 |
| RF-DETR-Medium | 18-20 GB | ~25 min on RTX 4090 |
| RT-DETR-L | 14-16 GB | ~20 min on T4 |

T4 (16 GB, free on Lightning.ai) handles everything except RF-DETR-Medium.

## Output

After training, the ONNX model is at:
```
runs/cryoppp-picker/<model_name>/weights/best.onnx
```

Copy this to the main Magellon repo at `models/` and test with:
```bash
cargo test --features "onnx,cryoppp-data" -- cryoppp
```
