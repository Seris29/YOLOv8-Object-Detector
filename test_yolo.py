from ultralytics import YOLO
import cv2
import glob
import os

# ---------------------------
# Locate the latest best.pt
# ---------------------------
weights_folder = "runs/detect/train3/weights"
best_models = glob.glob(os.path.join(weights_folder, "best*.pt"))
if not best_models:
    raise FileNotFoundError(f"No best.pt files found in {weights_folder}")
best_model_path = sorted(best_models)[-1]  # take the last one if multiple

print(f"Using model: {best_model_path}")

# ---------------------------
# Load your trained model
# ---------------------------
model = YOLO(best_model_path)

# ---------------------------
# Open webcam (0 = default Mac camera)
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam!")

# ---------------------------
# Class names (from data.yaml)
# ---------------------------
CLASS_NAMES = ["Robot-car", "ball", "goalpost"]

# ---------------------------
# Detection loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Annotated frame with boxes
    annotated_frame = results[0].plot()

    # Extract boxes info
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Class label
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]

            # Print detection info
            print(f"{label} -> center: ({int(x_center)}, {int(y_center)})")

            # Draw center point on frame
            cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)

    # Show live window
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
