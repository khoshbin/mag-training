#!/usr/bin/env python3
"""Export trained models to ONNX and validate against CryoPPP ground truth.

Supports RT-DETR and YOLOv11 models trained via Ultralytics.

Usage:
    # Export best.pt to ONNX
    python training/export_onnx.py --model runs/cryoppp-picker/rtdetr_10017/weights/best.pt \
        --imgsz 640 --format onnx

    # Export + validate against CryoPPP ground truth
    python training/export_onnx.py --model runs/cryoppp-picker/rtdetr_10017/weights/best.pt \
        --validate --data data/cryoppp/lite/10017 --box-size 108 --tolerance 20
"""

import argparse
import csv
import struct
from pathlib import Path


def export_model(model_path: str, imgsz: int, fmt: str = "onnx") -> str:
    """Export Ultralytics model to specified format."""
    from ultralytics import RTDETR, YOLO

    # Auto-detect model type
    model_path = Path(model_path)

    # Try RT-DETR first, fall back to YOLO
    try:
        model = RTDETR(str(model_path))
        print(f"Loaded RT-DETR model: {model_path}")
    except Exception:
        model = YOLO(str(model_path))
        print(f"Loaded YOLO model: {model_path}")

    # Export
    export_path = model.export(
        format=fmt,
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        half=False,
    )

    print(f"Exported to: {export_path}")

    # Report model size
    onnx_path = Path(export_path)
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")

    return str(export_path)


def parse_coordinates_csv(csv_path: Path) -> list[tuple[float, float]]:
    """Parse CryoPPP coordinate CSV."""
    particles = []
    with open(csv_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and any(c.isalpha() for c in line):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            if len(parts) >= 2:
                try:
                    particles.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return particles


def validate_onnx(
    onnx_path: str,
    data_dir: str,
    box_size: int,
    tolerance: float,
    imgsz: int,
):
    """Run ONNX model on CryoPPP data and report F1/precision/recall."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: pip install onnxruntime")
        return

    print(f"\n=== Validating ONNX model against CryoPPP ground truth ===")
    print(f"Model: {onnx_path}")
    print(f"Data:  {data_dir}")

    # Load ONNX model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name} shape={input_shape}")

    # Find micrographs and coordinates
    data_path = Path(data_dir)
    mic_dir = data_path / "micrographs"
    if not mic_dir.exists():
        mic_dir = data_path

    coord_dir = data_path / "ground_truth" / "particle_coordinates"
    if not coord_dir.exists():
        coord_dir = data_path / "ground_truth"

    # Index coordinates
    coord_map = {}
    if coord_dir.exists():
        for csv_file in coord_dir.glob("*.csv"):
            coord_map[csv_file.stem] = csv_file

    total_tp, total_fp, total_fn = 0, 0, 0
    processed = 0

    for img_path in sorted(mic_dir.iterdir()):
        ext = img_path.suffix.lower()
        if ext not in (".mrc", ".jpg", ".jpeg", ".png"):
            continue

        stem = img_path.stem
        csv_path = coord_map.get(stem)
        if csv_path is None:
            # Partial match
            for cs, cp in coord_map.items():
                if stem in cs or cs in stem:
                    csv_path = cp
                    break
        if csv_path is None:
            continue

        gt_particles = parse_coordinates_csv(csv_path)
        if not gt_particles:
            continue

        # Load image
        if ext == ".mrc":
            img = load_mrc_as_array(img_path)
        else:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)

        if img is None:
            continue

        orig_h, orig_w = img.shape

        # Normalize and resize
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / max(img_max - img_min, 1e-8)

        # Resize to model input
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
        img_resized = img_pil.resize((imgsz, imgsz))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # Stack to 3 channels and add batch dim: [1, 3, H, W]
        input_tensor = np.stack([img_array] * 3)[np.newaxis, ...]

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        detections = outputs[0]  # [1, N, 5+]

        if len(detections.shape) == 3:
            detections = detections[0]  # [N, 5+]

        # Filter by confidence
        scale_x = orig_w / imgsz
        scale_y = orig_h / imgsz
        predictions = []
        for det in detections:
            cx, cy, w, h = det[:4]
            score = det[4:].max() if len(det) > 4 else det[4] if len(det) > 4 else 0.0
            if score >= 0.3:
                predictions.append((cx * scale_x, cy * scale_y))

        # Evaluate
        tp, fp, fn = evaluate(predictions, gt_particles, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        processed += 1

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"  {stem}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} ({tp}TP {fp}FP {fn}FN)")

    # Summary
    if processed == 0:
        print("No images processed!")
        return

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"\n=== Overall Results ({processed} micrographs) ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")


def evaluate(
    predictions: list[tuple[float, float]],
    ground_truth: list[tuple[float, float]],
    tolerance: float,
) -> tuple[int, int, int]:
    """Greedy matching: returns (TP, FP, FN)."""
    gt_matched = [False] * len(ground_truth)
    tp = 0

    for px, py in predictions:
        best_dist = float("inf")
        best_idx = -1
        for i, (gx, gy) in enumerate(ground_truth):
            if gt_matched[i]:
                continue
            dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            gt_matched[best_idx] = True
            tp += 1

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def load_mrc_as_array(path: Path):
    """Load MRC file as 2D numpy array (first slice)."""
    import numpy as np

    try:
        import mrcfile
        with mrcfile.open(str(path), mode="r") as mrc:
            data = mrc.data
            if data.ndim == 3:
                data = data[0]
            return data.astype(np.float32)
    except ImportError:
        pass

    # Fallback: manual MRC header parsing
    try:
        with open(path, "rb") as f:
            nx, ny, nz, mode = struct.unpack("<4i", f.read(16))
            f.seek(1024)  # skip header
            if mode == 2:  # float32
                data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32)
                return data.reshape(ny, nx)
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Export and validate ONNX models")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--format", default="onnx", help="Export format (onnx, torchscript, etc.)")
    parser.add_argument("--validate", action="store_true", help="Validate against CryoPPP data")
    parser.add_argument("--data", help="CryoPPP data directory for validation")
    parser.add_argument("--box-size", type=int, default=108, help="Particle box size in pixels")
    parser.add_argument("--tolerance", type=float, default=20.0, help="Matching tolerance in pixels")
    args = parser.parse_args()

    # Export
    onnx_path = export_model(args.model, args.imgsz, args.format)

    # Validate
    if args.validate:
        if not args.data:
            print("ERROR: --data required for validation")
            return
        validate_onnx(onnx_path, args.data, args.box_size, args.tolerance, args.imgsz)


if __name__ == "__main__":
    main()
