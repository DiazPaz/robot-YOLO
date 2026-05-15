import cv2
from ultralytics import YOLO

MODEL_PATH = "runs\\detect\\runs_toy\\toy_yolo_v1-4\\weights\\best.pt"
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.75

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: no se pudo leer la cámara.")
        break

    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    frame_marked = results[0].plot()

    cv2.imshow("Prueba YOLO - Toy", frame_marked)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()