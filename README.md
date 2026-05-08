# YOLO Object Detection — Minimal Toolkit

This repository provides a small, practical set of utilities for creating and editing YOLO-format labels and for running a trained YOLOv8 model in real time.

**Remaining scripts**
- `label_images.py`: create full-image YOLO `.txt` labels for a chosen class.
- `edit_labels.py`: interactive editor to inspect and change class IDs in existing YOLO label files.
- `test_yolo.py`: run real-time detection using a trained Ultralytics YOLO model (webcam).

**Quick usage**

- Create full-image labels for all images in a folder:

```bash
python label_images.py --dir Wheels --class-id 0
```

- Only create labels where missing:

```bash
python label_images.py --dir Wheels --class-id 0 --unlabeled-only
```

- Interactive edit for a single folder of images:

```bash
python edit_labels.py --images path/to/images --labels path/to/labels
```

- Interactive edit for dataset splits (each split must contain `images/` and `labels/`):

```bash
python edit_labels.py --splits train valid test --name-filter rc_IMG
```

- Run live detection (ensure `runs/detect/.../weights/best*.pt` exists):

```bash
python test_yolo.py
```

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the runtime dependencies:

```bash
pip install ultralytics opencv-python pyyaml
```

## Notes

- Labels use YOLO text format: `class_id x_center y_center width height` (normalized coordinates).
- `test_yolo.py` expects trained weights under `runs/detect/.../weights/best*.pt` — it picks the latest matching file.
- I removed several older helper scripts and consolidated functionality into the three scripts above.

## Model weights

- Official model files (`yolov8n.pt`, `yolov8s.pt`, etc.) are large and are not tracked in this repository.
- This project contains local checkpoint files under `runs/detect/.../weights/` (your trained `best.pt` / `last.pt`).
- Notes on the two commonly found base weights:
	- `yolov8n.pt` — YOLOv8-nano (smallest, fastest, lower accuracy). This was used as the base for the `runs/detect/train` run (`runs/detect/train/args.yaml` shows `model: yolov8n.pt`).
	- `yolov8s.pt` — YOLOv8-small (larger, higher accuracy).

If you publish this repo, don't commit `.pt` files. Use the checkpoints under `runs/` for your own experiments and add `.pt` to `.gitignore` (already configured).

## Author

Sergio Ramirez