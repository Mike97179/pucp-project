"""
Configuración central — rutas, hiperparámetros y nombres de clase.

Todo lo que solía estar duplicado en la sección 1 de cada script 01/02/03
ahora vive aquí una sola vez.

Los identificadores están en inglés; todo lo impreso al usuario está en
español, que es el idioma de trabajo del proyecto.
"""

import os
import yaml


# ------------------------------------------------------------------
# Entorno y rutas
# ------------------------------------------------------------------
IN_COLAB  = 'COLAB_GPU' in os.environ or 'google.colab' in str(os.environ)
BASE_PATH = '/content' if IN_COLAB else os.getcwd()

DATASET_PATH  = os.path.join(BASE_PATH, 'dataset')
IMG_DIR       = os.path.join(DATASET_PATH, 'images')
LBL_DIR       = os.path.join(DATASET_PATH, 'labels')
YAML_PATH     = os.path.join(DATASET_PATH, 'data.yaml')

TRAIN_TXT     = os.path.join(DATASET_PATH, 'train.txt')
VAL_TXT       = os.path.join(DATASET_PATH, 'val.txt')
TEST_TXT      = os.path.join(DATASET_PATH, 'test.txt')
BENCHMARK_TXT = os.path.join(DATASET_PATH, 'benchmark.txt')

MODELS_DIR    = os.path.join(BASE_PATH, 'models')
MODELS_CSV    = os.path.join(MODELS_DIR, 'models.csv')
ARCHIVE_DIR   = os.path.join(MODELS_DIR, 'archive')
DATASET_STATE = os.path.join(MODELS_DIR, 'dataset_state.txt')

BENCHMARK_OUTPUT = os.path.join(BASE_PATH, 'benchmark_results')
MATRICES_OUTPUT  = os.path.join(BASE_PATH, 'confusion_matrices')

ANALYZE_INPUT    = os.path.join(BASE_PATH, 'analyze', 'input')
ANALYZE_OUTPUT   = os.path.join(BASE_PATH, 'analyze', 'output')

IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')


# ------------------------------------------------------------------
# Hiperparámetros por defecto
# ------------------------------------------------------------------
SEED        = 42
IMGSZ       = 640
BATCH       = 8
EPOCHS      = 70
PATIENCE    = 20
WORKERS     = 2
DEVICE      = 0

# AdamW se fija explícitamente: con optimizer='auto' Ultralytics ignora
# el lr0 que le pasamos. Eso invalidó las corridas lr_0.001 / lr_0.01 / lr_0.02.
OPTIMIZER   = 'AdamW'
LR0         = 0.002          # ganador del barrido de learning rate
COS_LR      = True

BASE_MODEL  = 'yolov8n-seg.pt'
SPLIT_RATIO = (0.70, 0.20, 0.10)


# ------------------------------------------------------------------
# Clases
# ------------------------------------------------------------------
def load_class_names(yaml_path=YAML_PATH):
    """
    Lee los nombres de clase desde data.yaml.

    Se leen del yaml en vez de estar hardcodeados para que el mapeo
    ID -> clase nunca se desincronice si el orden cambia.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    names = data.get('names')
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def matrix_labels(names):
    """Nombres de clase más la fila/columna extra 'fondo' que agrega Ultralytics."""
    return list(names) + ['fondo']