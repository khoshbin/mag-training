#!/usr/bin/env python3
"""Fine-tune RT-DETR on CryoPPP particle picking data.

RT-DETR (Real-Time DEtection TRansformer) is a hybrid CNN+transformer detector
that achieves SOTA speed/accuracy and exports cleanly to ONNX.

Prerequisites:
    pip install -r training/requirements.txt
    python training/cryoppp_to_coco.py --input data/cryoppp/lite/10017 \
        --output data/training/10017 --box-size 108

Usage:
    # Quick smoke test (Tier A — laptop, CPU)
    python training/train_rtdetr.py --data data/training/10017/dataset.yaml \
        --epochs 5 --imgsz 640 --device cpu

    # Real training (Tier B — test server, GPU)
    python training/train_rtdetr.py --data data/training/10017/dataset.yaml \
        --epochs 100 --imgsz 1280 --device 0

    # Production training (Tier C — A100)
    python training/train_rtdetr.py --data data/training/merged/dataset.yaml \
        --model rtdetr-l --epochs 200 --imgsz 1280 --device 0 --batch 16
"""

import argparse
from pathlib import Path


def train(args):
    from ultralytics import RTDETR

    # Model selection
    model_map = {
        "rtdetr-r50": "rtdetr-resnet50.yaml",
        "rtdetr-r101": "rtdetr-resnet101.yaml",
        "rtdetr-l": "rtdetr-l.yaml",
    }

    if args.model in model_map:
        model = RTDETR(model_map[args.model])
        print(f"Training from scratch: {args.model}")
    elif args.model.endswith(".pt"):
        model = RTDETR(args.model)
        print(f"Fine-tuning from checkpoint: {args.model}")
    else:
        # Default: RT-DETR-L pretrained on COCO
        model = RTDETR("rtdetr-l.pt")
        print("Fine-tuning RT-DETR-L (COCO pretrained)")

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/cryoppp-picker",
        name=f"rtdetr_{Path(args.data).parent.name}",
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # CryoEM-specific augmentations (micrographs are grayscale, can rotate freely)
        hsv_h=0.0,     # no hue augmentation (grayscale)
        hsv_s=0.0,     # no saturation
        hsv_v=0.2,     # slight brightness variation
        degrees=180.0,  # full rotation (particles have no preferred orientation)
        flipud=0.5,    # vertical flip
        fliplr=0.5,    # horizontal flip
        mosaic=0.0,    # disable mosaic (micrographs are large, mosaic hurts)
        scale=0.3,     # moderate scale augmentation
    )

    print(f"\nTraining complete. Best model: {results.save_dir / 'weights' / 'best.pt'}")

    # Export to ONNX
    if args.export_onnx:
        best_model = RTDETR(str(results.save_dir / "weights" / "best.pt"))
        onnx_path = best_model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
        )
        print(f"ONNX exported: {onnx_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR on CryoPPP data")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="rtdetr-l", help="Model variant or .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="0", help="Device: cpu, 0, 0,1, etc.")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after training")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
