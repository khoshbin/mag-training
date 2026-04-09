#!/usr/bin/env bash
# setup.sh — One-command setup for Magellon training on Lightning.ai or any GPU
#
# Usage: bash setup.sh [--skip-data] [--model yolo11n] [--epochs 10]
set -euo pipefail

MODEL="${1:-yolo11n}"
EPOCHS="${2:-10}"
SKIP_DATA=false

for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=true ;;
    esac
done

echo "============================================"
echo "  Magellon Training Setup"
echo "  Model: ${MODEL}  Epochs: ${EPOCHS}"
echo "============================================"
echo ""

# ── Step 1: Install dependencies ──
echo "[1/5] Installing dependencies..."
pip install -q ultralytics mrcfile opencv-python-headless numpy Pillow 2>/dev/null
# RF-DETR (optional, skip if not needed)
pip install -q rfdetr 2>/dev/null || echo "  Note: rfdetr not installed (install manually for RF-DETR training)"
echo "  Done."

# ── Step 2: Check GPU ──
echo ""
echo "[2/5] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {gpu} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected. Training will be slow on CPU.')
"

# ── Step 3: Download data ──
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "[3/5] Downloading CryoPPP EMPIAR-10017 lite (476 MB)..."
    bash fetch_data.sh
else
    echo ""
    echo "[3/5] Skipping data download (--skip-data)"
fi

# ── Step 4: Convert to YOLO format ──
echo ""
echo "[4/5] Converting dataset to YOLO format..."
python cryoppp_to_coco.py \
    --input data/cryoppp/lite/10017 \
    --output data/training/10017 \
    --box-size auto \
    --seed 42

# ── Step 5: Train ──
echo ""
echo "[5/5] Training ${MODEL} for ${EPOCHS} epochs..."
echo ""

case "$MODEL" in
    yolo11n|yolo11s|yolo11m|yolo11l)
        python train_yolov11.py \
            --data data/training/10017/dataset.yaml \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --device 0 \
            --export-onnx
        ;;
    rfdetr-base|rfdetr)
        python train_rfdetr.py \
            --data data/training/10017 \
            --model base \
            --epochs "$EPOCHS" \
            --device 0 \
            --export-onnx
        ;;
    rfdetr-medium)
        python train_rfdetr.py \
            --data data/training/10017 \
            --model medium \
            --epochs "$EPOCHS" \
            --device 0 \
            --export-onnx
        ;;
    rtdetr)
        python train_rtdetr.py \
            --data data/training/10017/dataset.yaml \
            --epochs "$EPOCHS" \
            --device 0 \
            --export-onnx
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        echo "Options: yolo11n, yolo11s, yolo11m, rfdetr-base, rfdetr-medium, rtdetr"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo ""
echo "Results:"
find runs/ -name "*.onnx" 2>/dev/null | while read f; do
    echo "  ONNX: $f ($(du -sh "$f" | cut -f1))"
done
find runs/ -name "best.pt" 2>/dev/null | while read f; do
    echo "  PyTorch: $f ($(du -sh "$f" | cut -f1))"
done
echo ""
echo "Next steps:"
echo "  1. Download the ONNX model from runs/cryoppp-picker/*/weights/best.onnx"
echo "  2. Copy to your Magellon repo: models/particle_picker.onnx"
echo "  3. Test: cargo test --features 'onnx,cryoppp-data' -- cryoppp"
