import argparse
import os
from pathlib import Path
import cv2

CLASS_NAMES = ['Robot-car', 'ball', 'goalpost']
COLORS = [(0,255,0), (0,165,255), (0,0,255)]

def read_label_file(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, bw, bh = map(float, parts)
            cls = int(cls)
            boxes.append([cls, x, y, bw, bh])
    return boxes

def write_label_file(label_path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")

def draw_yolo_boxes(image_path, label_path, class_names=CLASS_NAMES):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
    h, w = img.shape[:2]
    raw_boxes = read_label_file(label_path)
    if not raw_boxes:
        print(f"No label for {image_path}")
        return False

    boxes = []
    for b in raw_boxes:
        cls, x, y, bw, bh = b
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        boxes.append([cls, x, y, bw, bh, x1, y1, x2, y2])

    idx = 0
    changed = False
    while True:
        img_disp = img.copy()
        for i, box in enumerate(boxes):
            color = COLORS[int(box[0]) % len(COLORS)]
            cv2.rectangle(img_disp, (box[5], box[6]), (box[7], box[8]), color, 2)
            label = class_names[int(box[0])] if int(box[0]) < len(class_names) else str(int(box[0]))
            cv2.putText(img_disp, f"{label} ({int(box[0])})", (box[5], box[6]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if i == idx:
                cv2.rectangle(img_disp, (box[5], box[6]), (box[7], box[8]), (255,255,255), 2)

        cv2.imshow('Edit Labels', img_disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(boxes)
        elif key == ord('p'):
            idx = (idx - 1) % len(boxes)
        elif key in [ord(str(i)) for i in range(10)]:
            new_class = int(chr(key))
            boxes[idx][0] = new_class
            changed = True
            print(f"Changed box {idx+1} to class {new_class}")
        elif key == ord('s'):
            # Save updated labels
            out_boxes = [[b[0], b[1], b[2], b[3], b[4]] for b in boxes]
            write_label_file(label_path, out_boxes)
            print(f"Saved updated labels to {label_path}")
            changed = False
            break

    cv2.destroyAllWindows()
    return changed

def iter_images_labels(images_dir, labels_dir, name_filter=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    if not images_dir.exists():
        return []
    files = [f for f in images_dir.iterdir() if f.suffix.lower() in exts]
    if name_filter:
        files = [f for f in files if name_filter in f.name]
    files.sort()
    pairs = [(f, labels_dir / (f.stem + '.txt')) for f in files]
    return pairs

def main():
    p = argparse.ArgumentParser(description='Interactive YOLO label editor (single folder or splits).')
    p.add_argument('--images', help='Folder with images')
    p.add_argument('--labels', help='Folder with YOLO label files')
    p.add_argument('--splits', nargs='+', help='List of split folders (each should contain images/ and labels/)')
    p.add_argument('--name-filter', help='Optional substring to filter image filenames (used with --splits)')
    args = p.parse_args()

    if args.splits:
        for split in args.splits:
            images_dir = Path(split) / 'images'
            labels_dir = Path(split) / 'labels'
            pairs = iter_images_labels(images_dir, labels_dir, name_filter=args.name_filter)
            for img_path, label_path in pairs:
                print(f"Editing: {img_path.relative_to(images_dir.parent)}")
                draw_yolo_boxes(img_path, label_path)
    else:
        if not args.images or not args.labels:
            p.error('Either --splits or both --images and --labels must be provided')
        pairs = iter_images_labels(args.images, args.labels)
        for img_path, label_path in pairs:
            print(f"Editing: {img_path.name}")
            draw_yolo_boxes(img_path, label_path)

if __name__ == '__main__':
    main()
