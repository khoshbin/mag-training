#!/usr/bin/env python3
"""Convert CryoPPP dataset to COCO format for YOLO/RF-DETR/RT-DETR training.

CryoPPP layout:
    <empiar_id>/micrographs/*.{mrc,jpg}
    <empiar_id>/ground_truth/particle_coordinates/*.csv

COCO output:
    images/     — micrograph images (converted to PNG if MRC, CLAHE-enhanced)
    labels/     — YOLO-format .txt files (class cx cy w h, normalized)
    dataset.yaml — Ultralytics dataset config

Usage:
    python training/cryoppp_to_coco.py --input data/cryoppp/lite/10017 \
        --output data/training/10017 --split 0.8 --seed 42

    # Auto-detect box size from CSV metadata (Diameter column):
    python training/cryoppp_to_coco.py --input data/cryoppp/lite/10017 \
        --output data/training/10017 --box-size auto

    # Explicit box size override:
    python training/cryoppp_to_coco.py --input data/cryoppp/lite/10017 \
        --output data/training/10017 --box-size 108
"""

import argparse
import random
import struct
import shutil
from pathlib import Path

import numpy as np


def parse_coordinates_csv(csv_path: Path) -> tuple[list[tuple[float, float]], int | None]:
    """Parse CryoPPP coordinate CSV.

    Returns (particles, diameter_px) where diameter_px is read from the
    'Diameter' column if present, else None.
    """
    particles = []
    diameter_px = None
    diameter_col = None
    header_cols = None

    with open(csv_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Parse header
            if i == 0 and any(c.isalpha() for c in line):
                if "," in line:
                    header_cols = [c.strip() for c in line.split(",")]
                elif "\t" in line:
                    header_cols = [c.strip() for c in line.split("\t")]
                else:
                    header_cols = line.split()

                # Find diameter column
                for idx, col in enumerate(header_cols):
                    if col.lower() in ("diameter", "diameter_px", "box_size"):
                        diameter_col = idx
                        break
                continue

            # Parse data rows
            if "," in line:
                parts = line.split(",")
            elif "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()

            if len(parts) >= 2:
                try:
                    x, y = float(parts[0].strip()), float(parts[1].strip())
                    particles.append((x, y))

                    # Read diameter from first data row
                    if diameter_px is None and diameter_col is not None and len(parts) > diameter_col:
                        try:
                            diameter_px = int(float(parts[diameter_col].strip()))
                        except (ValueError, IndexError):
                            pass
                except ValueError:
                    continue

    return particles, diameter_px


def find_image_size(img_path: Path) -> tuple[int, int]:
    """Get image dimensions. Supports JPG/PNG via PIL, MRC via header."""
    ext = img_path.suffix.lower()
    if ext in (".jpg", ".jpeg", ".png"):
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    elif ext == ".mrc":
        with open(img_path, "rb") as f:
            nx, ny, nz = struct.unpack("<3i", f.read(12))
        return (nx, ny)
    else:
        raise ValueError(f"Unknown image format: {ext}")


def load_mrc_as_array(path: Path) -> np.ndarray | None:
    """Load MRC file as 2D float32 array."""
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
            f.seek(1024)
            if mode == 2:  # float32
                data = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32)
                return data.reshape(ny, nx)
            elif mode == 1:  # int16
                data = np.frombuffer(f.read(nx * ny * 2), dtype=np.int16)
                return data.reshape(ny, nx).astype(np.float32)
            elif mode == 6:  # uint16
                data = np.frombuffer(f.read(nx * ny * 2), dtype=np.uint16)
                return data.reshape(ny, nx).astype(np.float32)
    except Exception as e:
        print(f"  WARN: Failed to read MRC {path}: {e}")
    return None


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization.

    Cryo-EM micrographs have very low contrast — CLAHE dramatically improves
    particle visibility for both human inspection and model training.
    """
    try:
        import cv2
        # Normalize to 0-255 uint8
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-8:
            return img
        normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(normalized)
        return enhanced.astype(np.float32)
    except ImportError:
        # Fallback: simple percentile normalization
        p2, p98 = np.percentile(img, (2, 98))
        clipped = np.clip(img, p2, p98)
        img_min, img_max = clipped.min(), clipped.max()
        if img_max - img_min < 1e-8:
            return img
        return ((clipped - img_min) / (img_max - img_min) * 255).astype(np.float32)


def mrc_to_png(mrc_path: Path, png_path: Path, use_clahe: bool = True) -> bool:
    """Convert MRC micrograph to PNG with optional CLAHE enhancement."""
    from PIL import Image

    data = load_mrc_as_array(mrc_path)
    if data is None:
        return False

    if use_clahe:
        data = apply_clahe(data)
    else:
        # Simple normalization to 0-255
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin) * 255
        data = data.astype(np.float32)

    img = Image.fromarray(data.astype(np.uint8), mode="L")
    img.save(str(png_path))
    return True


def convert_entry(
    input_dir: Path,
    output_dir: Path,
    box_size_px: int | None,
    train_split: float,
    seed: int,
    use_clahe: bool,
):
    """Convert one CryoPPP entry to YOLO format."""
    # Find micrographs
    mic_dir = input_dir / "micrographs"
    if not mic_dir.exists():
        mic_dir = input_dir

    coord_dir = input_dir / "ground_truth" / "particle_coordinates"
    if not coord_dir.exists():
        coord_dir = input_dir / "ground_truth"
    if not coord_dir.exists():
        coord_dir = input_dir / "particle_coordinates"

    # Index coordinate files by stem
    coord_map = {}
    if coord_dir.exists():
        for csv_file in coord_dir.glob("*.csv"):
            coord_map[csv_file.stem] = csv_file

    # Create output dirs
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process each micrograph
    entries = []
    auto_box_sizes = []
    image_extensions = {".mrc", ".jpg", ".jpeg", ".png"}

    for img_path in sorted(mic_dir.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        stem = img_path.stem

        # Find matching coordinates
        csv_path = coord_map.get(stem)
        if csv_path is None:
            for csv_stem, cp in coord_map.items():
                if stem in csv_stem or csv_stem in stem:
                    csv_path = cp
                    break

        if csv_path is None:
            print(f"  SKIP {stem}: no coordinate file")
            continue

        particles, diameter = parse_coordinates_csv(csv_path)
        if not particles:
            print(f"  SKIP {stem}: no particles in CSV")
            continue

        if diameter is not None:
            auto_box_sizes.append(diameter)

        w, h = find_image_size(img_path)
        entries.append((img_path, particles, w, h, stem, diameter))

    if not entries:
        print("ERROR: No valid entries found")
        return

    # Determine box size
    if box_size_px is None:
        if auto_box_sizes:
            # Use median diameter from CSV metadata
            box_size_px = int(np.median(auto_box_sizes))
            print(f"Auto-detected box size: {box_size_px} px (median of {len(auto_box_sizes)} entries)")
        else:
            box_size_px = 128
            print(f"WARNING: No diameter metadata found. Using default box size: {box_size_px} px")

    # Shuffle with fixed seed for reproducible train/val split (avoids temporal bias)
    random.seed(seed)
    random.shuffle(entries)

    n_train = int(len(entries) * train_split)
    train_entries = entries[:n_train]
    val_entries = entries[n_train:]

    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}, Box size: {box_size_px} px")

    converted_mrc = 0
    for split, split_entries in [("train", train_entries), ("val", val_entries)]:
        for img_path, particles, w, h, stem, diameter in split_entries:
            ext = img_path.suffix.lower()
            # Use per-entry diameter if available, else global box size
            entry_box = diameter if diameter is not None else box_size_px

            # Copy/convert image
            if ext in (".jpg", ".jpeg", ".png"):
                dest = output_dir / "images" / split / img_path.name
                if use_clahe and ext in (".jpg", ".jpeg"):
                    # Apply CLAHE to JPG too (lite dataset has 8-bit JPG)
                    from PIL import Image
                    img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
                    enhanced = apply_clahe(img)
                    Image.fromarray(enhanced.astype(np.uint8), mode="L").save(str(dest))
                else:
                    shutil.copy2(img_path, dest)
            elif ext == ".mrc":
                # MRC → PNG conversion (P0 fix: Ultralytics cannot read MRC)
                dest = output_dir / "images" / split / f"{stem}.png"
                if mrc_to_png(img_path, dest, use_clahe=use_clahe):
                    converted_mrc += 1
                else:
                    print(f"  WARN: Failed to convert {stem}.mrc to PNG, skipping")
                    continue
            else:
                continue

            # Write YOLO label (class cx cy w h — normalized)
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            with open(label_path, "w") as f:
                for px, py in particles:
                    cx_norm = px / w
                    cy_norm = py / h
                    w_norm = entry_box / w
                    h_norm = entry_box / h

                    # Clamp to [0, 1]
                    cx_norm = max(0.0, min(1.0, cx_norm))
                    cy_norm = max(0.0, min(1.0, cy_norm))

                    f.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    # Write dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['particle']\n")

    total_particles = sum(len(p) for _, p, _, _, _, _ in entries)
    print(f"Dataset YAML: {yaml_path}")
    print(f"Total images: {len(entries)}, MRC→PNG converted: {converted_mrc}")
    print(f"Total particles: {total_particles}, Box size: {box_size_px} px")
    if auto_box_sizes:
        print(f"Per-entry diameters: min={min(auto_box_sizes)}, max={max(auto_box_sizes)}, median={int(np.median(auto_box_sizes))}")


def main():
    parser = argparse.ArgumentParser(description="Convert CryoPPP to COCO/YOLO format")
    parser.add_argument("--input", required=True, help="CryoPPP entry directory")
    parser.add_argument("--output", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--box-size", default="auto",
                        help="Particle box size in pixels, or 'auto' to read from CSV metadata")
    parser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    args = parser.parse_args()

    box_size = None if args.box_size == "auto" else int(args.box_size)

    convert_entry(
        Path(args.input),
        Path(args.output),
        box_size,
        args.split,
        args.seed,
        use_clahe=not args.no_clahe,
    )


if __name__ == "__main__":
    main()
