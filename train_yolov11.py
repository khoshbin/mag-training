#!/usr/bin/env python3
"""Fine-tune YOLOv11 on CryoPPP particle picking data.

YOLOv11 is the fallback picker — faster to train, smaller models, proven in
Sandbox 01 (hole detection F1≈0.76). Use this when RT-DETR doesn't converge
on small datasets or when inference speed is critical.

Prerequisites:
    pip install -r training/requirements.txt
    python training/cryoppp_to_coco.py --input data/cryoppp/lite/10017 \
        --output data/training/10017 --box-size 108

Usage:
    # Smoke test (Tier A — laptop, CPU, nano model)
    python training/train_yolov11.py --data data/training/10017/dataset.yaml \
        --model yolo11n --epochs 10 --device cpu

    # Real training (Tier B — test server, GPU)
    python training/train_yolov11.py --data data/training/10017/dataset.yaml \
        --model yolo11s --epochs 100 --device 0

    # Production (Tier C — A100, larger model)
    python training/train_yolov11.py --data data/training/merged/dataset.yaml \
        --model yolo11m --epochs 200 --device 0 --batch 32
"""

import argparse
from pathlib import Path


def train(args):
    from ultralytics import YOLO

    # Model variants (nano → x-large)
    model_map = {
        "yolo11n": "yolo11n.pt",   # 2.6M params — fastest, for smoke tests
        "yolo11s": "yolo11s.pt",   # 9.4M params — good balance
        "yolo11m": "yolo11m.pt",   # 20.1M params — better accuracy
        "yolo11l": "yolo11l.pt",   # 25.3M params
        "yolo11x": "yolo11x.pt",   # 56.9M params — highest accuracy
    }

    if args.model in model_map:
        model = YOLO(model_map[args.model])
        print(f"Fine-tuning {args.model} (COCO pretrained)")
    elif args.model.endswith(".pt"):
        model = YOLO(args.model)
        print(f"Resuming from checkpoint: {args.model}")
    else:
        model = YOLO("yolo11n.pt")
        print("Defaulting to yolo11n")

    # Train with cryo-EM specific augmentations
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/cryoppp-picker",
        name=f"yolo11_{Path(args.data).parent.name}",
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # CryoEM-specific augmentations
        hsv_h=0.0,     # grayscale — no hue
        hsv_s=0.0,     # grayscale — no saturation
        hsv_v=0.2,     # slight brightness variation
        degrees=180.0,  # full rotation (no preferred particle orientation)
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,    # disable mosaic (micrographs are large, mosaic hurts)
        scale=0.3,
        translate=0.1,
    )

    print(f"\nTraining complete. Best model: {results.save_dir / 'weights' / 'best.pt'}")

    # Export to ONNX
    if args.export_onnx:
        best = YOLO(str(results.save_dir / "weights" / "best.pt"))
        onnx_path = best.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
        )
        print(f"ONNX exported: {onnx_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on CryoPPP data")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo11n", help="Model variant or .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Device: cpu, 0, 0,1, etc.")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after training")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
