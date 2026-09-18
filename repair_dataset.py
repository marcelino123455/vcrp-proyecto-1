from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent / "data"
BAD_DIR = ROOT / "bad"
DATASETS = ["oxford", "paris"]


def is_probably_html(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(512)
    except OSError:
        return True
    text = head.lower()
    return b"<html" in text or b"302 found" in text or b"doctype html" in text or b"<script" in text


def main() -> None:
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for dataset in DATASETS:
        image_dir = ROOT / dataset
        if not image_dir.exists():
            print(f"No existe {image_dir}, se omite.")
            continue

        bad_dataset_dir = BAD_DIR / dataset
        bad_dataset_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(image_dir.glob("*.jpg")):
            if is_probably_html(image_path):
                target = bad_dataset_dir / image_path.name
                if target.exists():
                    target.unlink()
                image_path.rename(target)
                print(f"Movido a respaldo (HTML): {image_path}")
                removed += 1
                continue

            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                target = bad_dataset_dir / image_path.name
                if target.exists():
                    target.unlink()
                image_path.rename(target)
                print(f"Movido a respaldo (no legible): {image_path}")
                removed += 1

    print(f"Total de archivos movidos a respaldo: {removed}")


if __name__ == "__main__":
    main()
