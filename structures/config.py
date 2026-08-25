"""
Central configuration — paths, hyperparameters and class names.

Everything that used to be duplicated in section 1 of every 01/02/03 script
lives here once.

Note on language: identifiers, docstrings, comments and the command line
surface are in English; every string printed to the user stays in Spanish,
since that is the project's working language.
"""

import os
import yaml


# ------------------------------------------------------------------
# Environment and paths
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
HISTORY_DIR   = os.path.join(MODELS_DIR, 'history')
ARCHIVE_DIR   = os.path.join(MODELS_DIR, 'archive')
DATASET_STATE = os.path.join(MODELS_DIR, 'dataset_state.txt')

BENCHMARK_OUTPUT = os.path.join(BASE_PATH, 'benchmark_results')
MATRICES_OUTPUT  = os.path.join(BASE_PATH, 'confusion_matrices')

IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')


# ------------------------------------------------------------------
# Default hyperparameters
# ------------------------------------------------------------------
SEED        = 42
IMGSZ       = 640
BATCH       = 8
EPOCHS      = 70
PATIENCE    = 20
WORKERS     = 2
DEVICE      = 0

# AdamW is pinned: with optimizer='auto' Ultralytics ignores the lr0 we pass.
# That default is what invalidated the lr_0.001 / lr_0.01 / lr_0.02 runs.
OPTIMIZER   = 'AdamW'
LR0         = 0.002          # winner of the learning rate sweep
COS_LR      = True

BASE_MODEL  = 'yolov8n-seg.pt'
SPLIT_RATIO = (0.70, 0.20, 0.10)


# ------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------
def load_class_names(yaml_path=YAML_PATH):
    """
    Read the class names from data.yaml.

    They are read from the yaml instead of being hardcoded so the ID -> class
    mapping never drifts if the order ever changes.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    names = data.get('names')
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def matrix_labels(names):
    """Class names plus the extra 'fondo' row/column Ultralytics adds."""
    return list(names) + ['fondo']
