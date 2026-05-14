import cv2
import os
import random

# =========================
# CONFIGURACIÓN
# =========================
video_path = "robot-YOLO\\frame_extractor\\toy_videos\\toy2.mp4"
output_dir = "robot-YOLO\\frames_random"
num_frames_to_extract = 200
seed = 42   # cámbialo o quítalo si no quieres reproducibilidad

# =========================
# PREPARAR SALIDA
# =========================
os.makedirs(output_dir, exist_ok=True)
random.seed(seed)

# =========================
# ABRIR VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"No se pudo abrir el video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames == 0:
    cap.release()
    raise ValueError("El video no contiene frames o no pudo leerse correctamente.")

# Si el video tiene menos de 576 frames, toma todos los disponibles
n = min(num_frames_to_extract, total_frames)

# Elegir índices aleatorios únicos, sin orden particular
selected_indices = random.sample(range(total_frames), n)
random.shuffle(selected_indices)  # opcional, para reforzar el desorden

# =========================
# EXTRAER Y GUARDAR
# =========================
saved = 0

for i, frame_idx in enumerate(selected_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"No se pudo leer el frame {frame_idx}, se omite.")
        continue

    # Guarda con el índice original del video
    filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(filename, frame)
    saved += 1

cap.release()

print(f"Frames guardados: {saved}")
print(f"Carpeta de salida: {output_dir}")