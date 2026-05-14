from pathlib import Path
from ultralytics import YOLO
import torch


# =========================
# CONFIGURACIÓN BÁSICA
# =========================

DATA_YAML = "robot-YOLO\\toy_dataset\\data.yaml"          # Archivo .yaml de tu dataset
BASE_MODEL = "yolo11n.pt"        # Modelo base ligero para entrenar rápido
EPOCHS = 80                      # Número de épocas de entrenamiento
IMG_SIZE = 640                   # Tamaño de imagen para YOLO
BATCH_SIZE = 8                   # Baja este valor si tu PC se queda sin memoria

PROJECT_NAME = "runs_toy"
RUN_NAME = "toy_yolo_v1"


def main():
    # =========================
    # 1. VERIFICAR ARCHIVOS
    # =========================

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"No se encontró el archivo {DATA_YAML}")

    if not Path("robot-YOLO\\toy_dataset\\train.txt").exists():
        raise FileNotFoundError("No se encontró train.txt")

    # =========================
    # 2. SELECCIONAR CPU O GPU
    # =========================

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en: {device}")

    # =========================
    # 3. CARGAR MODELO BASE
    # =========================

    model = YOLO(BASE_MODEL)

    # =========================
    # 4. ENTRENAR MODELO
    # =========================

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=20,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        workers=0
    )

    # =========================
    # 5. UBICAR MODELO ENTRENADO
    # =========================

    best_model_path = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    last_model_path = Path(PROJECT_NAME) / RUN_NAME / "weights" / "last.pt"

    print("\nEntrenamiento terminado.")
    print(f"Mejor modelo: {best_model_path}")
    print(f"Último modelo: {last_model_path}")

    # =========================
    # 6. VALIDAR MODELO ENTRENADO
    # =========================

    trained_model = YOLO(str(best_model_path))
    trained_model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=device
    )

    # =========================
    # 7. HACER UNA PREDICCIÓN DE PRUEBA
    # =========================

    with open("robot-YOLO\\toy_dataset\\train.txt", "r", encoding="utf-8") as file:
        first_image = file.readline().strip()

    if first_image:
        print(f"\nProbando detección con: {first_image}")
        trained_model.predict(
            source=first_image,
            conf=0.5,
            save=True
        )

    print("\nListo. Revisa la carpeta de resultados generada.")


if __name__ == "__main__":
    main()