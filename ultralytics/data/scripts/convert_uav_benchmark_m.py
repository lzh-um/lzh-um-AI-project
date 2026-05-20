# Ultralytics \ud83d\ude80 AGPL-3.0 License - https://ultralytics.com/license
"""Utility for converting the UAV-benchmark-M dataset into YOLO-ready structure."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from ultralytics.utils import LOGGER, TQDM

CLASS_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["car", "truck", "bus"]
IMAGE_GLOB = "*.jpg"


def load_sequences(attr_dir: Path) -> List[str]:
    """Return sorted sequence identifiers listed inside an attribute split directory."""
    if not attr_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {attr_dir}")
    seqs = sorted(p.stem.split("_", 1)[0] for p in attr_dir.glob("M*_attr.txt"))
    if not seqs:
        raise FileNotFoundError(f"No sequence attribute files found under {attr_dir}")
    return seqs


def read_annotations(gt_file: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Parse a *_gt_whole.txt file into per-frame annotations."""
    boxes: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    with gt_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                frame = int(float(row[0]))
                left, top, width, height = map(float, row[2:6])
                raw_class = int(float(row[8]))
            except ValueError:
                continue
            cls_id = CLASS_MAP.get(raw_class)
            if cls_id is None or width <= 1 or height <= 1:
                continue
            boxes[frame].append((cls_id, left, top, width, height))
    return boxes


def materialize_image(src: Path, dst: Path, copy_images: bool) -> None:
    """Link or copy a source image into the YOLO dataset structure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError as exc:  # fallback when symlinks are unavailable
            LOGGER.warning(f"Symlink failed for {dst.name}: {exc}. Copying instead.")
            shutil.copy2(src, dst)


def clamp_bbox(left: float, top: float, width: float, height: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Clamp a bounding box to the image bounds and return xywh."""
    x1 = max(0.0, left)
    y1 = max(0.0, top)
    x2 = min(float(img_w), left + width)
    y2 = min(float(img_h), top + height)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h


def convert_sequence(
    source_root: Path,
    seq: str,
    split: str,
    gt_root: Path,
    images_dir: Path,
    labels_dir: Path,
    copy_images: bool,
) -> Tuple[int, int]:
    """Convert a single UAV sequence into YOLO images/labels and return counts."""
    seq_dir = source_root / seq
    gt_file = gt_root / f"{seq}_gt_whole.txt"
    if not seq_dir.exists():
        LOGGER.warning(f"Skipping {seq}: missing image folder {seq_dir}")
        return 0, 0
    if not gt_file.exists():
        LOGGER.warning(f"Skipping {seq}: missing ground-truth file {gt_file}")
        return 0, 0

    annotations = read_annotations(gt_file)
    image_files = sorted(seq_dir.glob(IMAGE_GLOB))
    if not image_files:
        LOGGER.warning(f"Skipping {seq}: no images found")
        return 0, 0

    split_image_dir = images_dir / split
    split_label_dir = labels_dir / split
    split_label_dir.mkdir(parents=True, exist_ok=True)

    written_labels = 0
    for img_path in image_files:
        stem = img_path.stem
        try:
            frame_idx = int(stem.split("img")[-1])
        except ValueError:
            LOGGER.warning(f"Unable to parse frame index from {img_path.name}; skipping")
            continue
        dest_stem = f"{seq}_{stem}"
        dest_image = split_image_dir / f"{dest_stem}{img_path.suffix.lower()}"
        materialize_image(img_path, dest_image, copy_images)

        frame_boxes = annotations.get(frame_idx, [])
        label_path = split_label_dir / f"{dest_stem}.txt"
        if not frame_boxes:
            label_path.write_text("", encoding="utf-8")
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lines = []
        for cls_id, left, top, width, height in frame_boxes:
            cx, cy, w, h = clamp_bbox(left, top, width, height, img_w, img_h)
            if w <= 0 or h <= 0:
                continue
            lines.append(
                f"{cls_id} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"
            )
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written_labels += len(lines)

    return len(image_files), written_labels


def convert_uav_benchmark_m(
    source: Path,
    target: Path,
    val_ratio: float = 0.1,
    copy_images: bool = False,
    clean: bool = False,
) -> None:
    """Entry point for converting UAV-benchmark-M into YOLO layout."""
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    attr_root = source / "M_attr"
    train_attr = attr_root / "train"
    test_attr = attr_root / "test"
    gt_root = source / "UAV-benchmark-MOTD_v1.0" / "GT"
    if not gt_root.exists():
        raise FileNotFoundError(f"Missing GT directory: {gt_root}")

    train_sequences = load_sequences(train_attr)
    test_sequences = load_sequences(test_attr) if test_attr.exists() else []

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    val_count = int(round(len(train_sequences) * val_ratio))
    if val_ratio > 0:
        val_count = max(1, min(val_count, len(train_sequences) - 1))
    else:
        val_count = 0

    if val_count:
        val_sequences = train_sequences[-val_count:]
        train_sequences = train_sequences[:-val_count]
    else:
        val_sequences = []

    if not train_sequences:
        raise ValueError("Validation split consumed all training sequences; lower val_ratio.")
    if not val_sequences:
        LOGGER.warning("Validation split is empty; consider setting --val-ratio > 0 for YOLO training.")

    if clean and target.exists():
        shutil.rmtree(target)
    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "labels").mkdir(parents=True, exist_ok=True)

    splits: Dict[str, Sequence[str]] = {
        "train": train_sequences,
        "val": val_sequences,
        "test": test_sequences,
    }

    LOGGER.info(
        f"Prepared splits -> train: {len(train_sequences)}, val: {len(val_sequences)}, test: {len(test_sequences)}"
    )

    stats = {}
    for split, sequences in splits.items():
        if not sequences:
            continue
        LOGGER.info(f"Converting {split} split with {len(sequences)} sequences")
        total_images = 0
        total_labels = 0
        for seq in TQDM(sequences, desc=f"{split} sequences"):
            images_written, labels_written = convert_sequence(
                source, seq, split, gt_root, target / "images", target / "labels", copy_images
            )
            total_images += images_written
            total_labels += labels_written
        stats[split] = (total_images, total_labels)
        LOGGER.info(f"{split}: {total_images} images, {total_labels} boxes")

    meta_lines = ["Dataset conversion finished:"]
    for split, (img_count, box_count) in stats.items():
        meta_lines.append(f"  - {split}: {img_count} images, {box_count} objects")
    meta_lines.append(f"Classes: {CLASS_NAMES}")
    (target / "uav_benchmark_m_meta.txt").write_text("\n".join(meta_lines), encoding="utf-8")
    LOGGER.info("\n".join(meta_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UAV-benchmark-M to YOLO format.")
    parser.add_argument("--source", type=Path, required=True, help="Path to the extracted UAV-benchmark-M directory")
    parser.add_argument(
        "--target",
        type=Path,
        help="Output directory for the YOLO-ready dataset (default: <source>_YOLO)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training sequences reserved for validation (0-1)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of using lightweight symlinks",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing target directory before conversion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target or (args.source.expanduser().resolve().parent / "UAV-benchmark-M-YOLO")
    convert_uav_benchmark_m(
        source=args.source,
        target=target,
        val_ratio=args.val_ratio,
        copy_images=args.copy,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
