#!/usr/bin/env python3
"""Create YOLO-format labels for images as a single chosen class.

Usage examples:
  python label_images.py --dir Wheels --class-id 0
  python label_images.py --dir dataset/images --class-id 2 --unlabeled-only
  python label_images.py --dir imgs --class-id 1 --overwrite
"""
from pathlib import Path
import argparse
import sys

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_full_image_label(img_path: Path, class_id: int, overwrite: bool) -> bool:
    txt = img_path.with_suffix('.txt')
    if txt.exists() and not overwrite:
        return False
    txt.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding='utf-8')
    return True


def main() -> None:
    p = argparse.ArgumentParser(description='Create full-image YOLO labels for every image in a folder.')
    p.add_argument('--dir', required=True, help='Folder containing images')
    p.add_argument('--class-id', required=True, type=int, help='Class id to write into labels')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing label files')
    p.add_argument('--unlabeled-only', action='store_true', help='Only create labels for images missing a .txt file')
    args = p.parse_args()

    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        print(f"Directory not found: {root}")
        sys.exit(2)

    imgs = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not imgs:
        print("No images found.")
        return

    created = 0
    skipped = 0
    for img in imgs:
        txt = img.with_suffix('.txt')
        if args.unlabeled_only and txt.exists():
            skipped += 1
            continue
        ok = create_full_image_label(img, args.class_id, args.overwrite)
        if ok:
            created += 1
        else:
            skipped += 1

    print(f"Images processed: {len(imgs)}, labels created: {created}, skipped: {skipped}")


if __name__ == '__main__':
    main()