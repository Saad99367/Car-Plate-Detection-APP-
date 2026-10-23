
import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
BEST_MODEL_PATH = r"CarPlateYOLO.pt"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.40
IMAGE_SIZE = 640

# ---------------- LOAD MODEL ----------------
if not Path(BEST_MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Model not found: {BEST_MODEL_PATH}\n"
        "Update BEST_MODEL_PATH to your actual best.pt file."
    )

model = YOLO(BEST_MODEL_PATH)

# ---------------- OPEN WEBCAM ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Check camera permissions "
        "or try CAMERA_INDEX = 1."
    )

# Optional camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("License plate detection started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read webcam frame.")
            break

        # Run YOLO detection
        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        # Draw bounding boxes and confidence scores
        annotated_frame = results[0].plot()

        cv2.imshow(
            "YOLO License Plate Detection",
            annotated_frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
