#!/usr/bin/env python3
"""Fine-tune RF-DETR on CryoPPP particle picking data.

RF-DETR (Roboflow, ICLR 2026) is the primary picker model — first real-time
detector to break 60 mAP on COCO. Built on DINOv2 backbone with deformable
attention. Designed for fine-tuning on custom datasets.

Prerequisites:
    pip install rfdetr
    python training/cryoppp_to_coco.py --input data/cryoppp/lite/10017 \
        --output data/training/10017 --box-size auto

Usage:
    # Quick smoke test (Tier A — laptop, CPU)
    python training/train_rfdetr.py --data data/training/10017 \
        --epochs 5 --device cpu --model base

    # Real training (Tier B — test server, GPU)
    python training/train_rfdetr.py --data data/training/10017 \
        --epochs 100 --device 0 --model medium

    # Production training (Tier C — A100)
    python training/train_rfdetr.py --data data/training/merged \
        --model large --epochs 200 --device 0 --batch 16
"""

import argparse
from pathlib import Path


def train(args):
    from rfdetr import RFDETRBase, RFDETRMedium, RFDETRLarge

    # Model selection
    model_map = {
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }

    model_cls = model_map.get(args.model)
    if model_cls is None:
        print(f"Unknown model: {args.model}. Options: base, medium, large")
        return

    model = model_cls()
    print(f"Training RF-DETR-{args.model} on {args.data}")

    # RF-DETR uses a different training API than Ultralytics
    # It expects COCO-format dataset directory with train/val splits
    model.train(
        dataset_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=max(1, 16 // args.batch),  # effective batch ~16
        lr=args.lr,
        output_dir=f"runs/cryoppp-picker/rfdetr_{args.model}_{Path(args.data).name}",
    )

    print(f"\nTraining complete.")

    # Export to ONNX
    if args.export_onnx:
        onnx_path = model.export()
        print(f"ONNX exported: {onnx_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR on CryoPPP data")
    parser.add_argument("--data", required=True, help="Path to COCO dataset directory")
    parser.add_argument("--model", default="medium",
                        choices=["base", "medium", "large"],
                        help="RF-DETR model size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="0", help="Device: cpu, 0, 0,1, etc.")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after training")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
