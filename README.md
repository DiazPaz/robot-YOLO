# 🤖 Robot Vision — Detección de Objetos con YOLO

Sistema de visión computacional integrado a un robot móvil autónomo, capaz de detectar, seguir y registrar una pieza específica en tiempo real mediante un modelo YOLO entrenado con dataset propio.

---

## 📋 Descripción General

Este repositorio contiene el pipeline completo de visión computacional para un robot autónomo: desde la generación del dataset y el entrenamiento del modelo, hasta la detección en tiempo real a través de la cámara del robot.

El sistema cumple con los siguientes objetivos:

- Detectar una pieza específica (etiquetada como `Toy`) usando un modelo YOLO personalizado.
- Mantener el seguimiento visual del objeto en tiempo real con bounding box actualizado.
- Utilizar una **Región de Interés (ROI)** para optimizar el procesamiento y reducir ruido visual.
- Registrar automáticamente cada evento de detección (número, timestamp, coordenadas, imagen).
- Almacenar los registros y evidencias en la nube (Google Drive).
- Operar de forma autónoma y estable durante al menos 7 minutos continuos.

---

## 🗂️ Estructura del Repositorio

```
robot-YOLO/
│
├── frame_extractor/
│   └── toy_videos/            # Videos fuente para extracción de frames
│
├── frames_random/             # Frames extraídos (generados por frame_extractor.py)
│
├── runs/
│   └── detect/
│       └── runs_toy/
│           └── toy_yolo_v1-4/
│               └── weights/
│                   └── best.pt    # Pesos del modelo entrenado ✅
│
├── frame_extractor.py         # Extrae frames aleatorios de videos para el dataset
├── train_yolo.ipynb           # Notebook de entrenamiento del modelo YOLO
├── main.py                    # Detección en tiempo real con la cámara
└── README.md
```

---

## 🔄 Flujo del Proyecto

```
Videos del objeto
       │
       ▼
frame_extractor.py   →   frames_random/   →   Etiquetado manual (LabelImg / Roboflow)
                                                        │
                                                        ▼
                                               toy_dataset/ (en Google Drive)
                                                        │
                                                        ▼
                                              train_yolo.ipynb   →   best.pt
                                                                          │
                                                                          ▼
                                                                       main.py
                                                                    (detección live)
```

---

## 📁 Descripción de Archivos

### `frame_extractor.py`
Extrae `N` frames aleatorios de un video fuente para construir el dataset de entrenamiento.

**Funcionamiento:**
1. Abre el video definido en `video_path`.
2. Calcula el total de frames disponibles.
3. Selecciona aleatoriamente `num_frames_to_extract` índices únicos (con semilla reproducible).
4. Guarda cada frame como `.jpg` en `output_dir` con el nombre `frame_XXXXXX.jpg`.

**Parámetros configurables:**
| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `video_path` | Ruta al video fuente | `toy_videos/toy2.mp4` |
| `output_dir` | Carpeta de salida de frames | `frames_random/` |
| `num_frames_to_extract` | Número de frames a extraer | `200` |
| `seed` | Semilla aleatoria para reproducibilidad | `42` |

**Uso:**
```bash
python frame_extractor.py
```

---

### `train_yolo.ipynb`
Notebook de entrenamiento del modelo YOLO11n sobre el dataset personalizado. Fue ejecutado en **VSCode** con los archivos del dataset alojados en **Google Drive**.

**Pipeline del notebook:**
1. **Instalación de dependencias** — `ultralytics`, `torch`.
2. **Montaje de Google Drive** — accede a `toy_dataset/` con el archivo `data.yaml` y `train.txt`.
3. **Validación del dataset** — verifica que todos los paths de imágenes en `train.txt` existen.
4. **Generación/actualización de `data.yaml`** — configura la clase `Toy` (id: `0`), rutas de train/val.
5. **Entrenamiento** — lanza `model.train()` con los parámetros:

| Parámetro | Valor |
|---|---|
| Modelo base | `yolo11n.pt` (nano, ligero) |
| Épocas | `80` |
| Tamaño de imagen | `640 × 640` |
| Batch size | `8` |
| Patience (early stopping) | `20` |
| Proyecto de salida | `runs_toy/toy_yolo_v1` |

**Requisitos para re-ejecutar:**
- Tener el dataset en Google Drive bajo la ruta `MyDrive/toy_dataset/`.
- El dataset debe incluir imágenes etiquetadas y un archivo `train.txt` con los paths.
- Instalar dependencias: `pip install ultralytics`.

---

### `best.pt`
Pesos del modelo YOLO entrenado, generados al finalizar `train_yolo.ipynb`. Contiene la configuración y los parámetros del mejor checkpoint obtenido durante el entrenamiento.

- **Clase detectada:** `Toy` (id `0`)
- **Arquitectura base:** YOLOv11 Nano (`yolo11n`)
- **Uso:** cargado directamente por `main.py` para inferencia en tiempo real.

---

### `main.py`
Script principal de detección en tiempo real. Captura video de la cámara del robot, aplica el modelo YOLO y visualiza los resultados en pantalla.

**Funcionamiento:**
1. Carga el modelo desde `best.pt`.
2. Abre la cámara en el índice `CAMERA_INDEX`.
3. Por cada frame capturado, ejecuta `model.predict()` con umbral de confianza `CONF_THRESHOLD`.
4. Dibuja los bounding boxes sobre el frame usando `results[0].plot()`.
5. Muestra el frame anotado en una ventana en tiempo real.
6. Presionar `q` cierra la aplicación limpiamente.

**Parámetros configurables:**
| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `MODEL_PATH` | Ruta al archivo de pesos entrenados | `runs/.../best.pt` |
| `CAMERA_INDEX` | Índice de la cámara (0 = integrada, 1 = externa) | `1` |
| `CONF_THRESHOLD` | Umbral mínimo de confianza para mostrar detección | `0.75` |

**Uso:**
```bash
python main.py
```

> **Nota:** Asegúrate de que `MODEL_PATH` apunta correctamente al archivo `best.pt` en tu sistema antes de ejecutar.

---

## ⚙️ Instalación y Requisitos

### Dependencias
```bash
pip install ultralytics opencv-python
```

### Versiones recomendadas
| Librería | Versión |
|---|---|
| Python | ≥ 3.9 |
| ultralytics | ≥ 8.0 |
| opencv-python | ≥ 4.8 |
| torch | ≥ 2.0 (con CUDA opcional) |

---

## 🚀 Guía de Uso Rápido

### 1. Generar dataset desde video
```bash
# Edita los paths en frame_extractor.py según tu entorno
python frame_extractor.py
```

### 2. Etiquetar imágenes
Utiliza una herramienta de etiquetado como [LabelImg](https://github.com/heartexlabs/labelImg) o [Roboflow](https://roboflow.com/) para anotar los frames extraídos con la clase `Toy`.

### 3. Entrenar el modelo
Abre `train_yolo.ipynb` en VSCode (con extensión Jupyter) o en Google Colab. Asegúrate de que el dataset esté disponible en Google Drive y ejecuta las celdas en orden.

### 4. Ejecutar detección en tiempo real
```bash
python main.py
```
Presiona `q` para salir.

---

## 🎯 Características del Sistema (Requerimientos del Proyecto)

| Requerimiento | Estado | Implementación |
|---|---|---|
| Entrenamiento de modelo YOLO propio | ✅ | `train_yolo.ipynb` + `best.pt` |
| Detección funcional de la pieza | ✅ | `main.py` con `CONF_THRESHOLD = 0.75` |
| Bounding box y seguimiento en tiempo real | ✅ | `results[0].plot()` en cada frame |
| Región de Interés (ROI) | 🔧 | Pendiente de implementar en `main.py` |
| Registro de eventos de detección | 🔧 | Pendiente (CSV/JSON con timestamp, coords, imagen) |
| Almacenamiento en la nube | 🔧 | Pendiente (Google Drive / servidor propio) |
| Operación autónoma ≥ 7 minutos | ✅ | Loop continuo en `main.py` |

---

## 📌 Notas

- El modelo fue entrenado con la clase `Toy` (índice `0`). Si se desean detectar otras clases, es necesario re-etiquetar y re-entrenar.
- El `CAMERA_INDEX = 1` asume una cámara USB externa. Cámbialo a `0` si usas la cámara integrada del equipo.
- Para mayor precisión en entornos reales, se recomienda aumentar el dataset con imágenes tomadas en las mismas condiciones de iluminación y fondo del escenario de prueba.