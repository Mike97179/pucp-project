"""
Training — a single wrapper around YOLO.train().

The three experiments (lr sweep, convergence, final model) all call the same
function; only the hyperparameters change.
"""

import os

import torch
from ultralytics import YOLO

from . import config


def check_environment():
    """Report whether a GPU is available. Never aborts: it just runs slower."""
    has_cuda = torch.cuda.is_available()
    print(f'CUDA disponible: {has_cuda}')
    if has_cuda:
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Sin GPU — funcionará igual pero será lento')
    return has_cuda


def next_run_dir(base=config.BASE_PATH, prefix='runs_final'):
    """First free name: runs_final_1, runs_final_2... without overwriting."""
    i = 1
    while os.path.exists(os.path.join(base, f'{prefix}_{i}')):
        i += 1
    return os.path.join(base, f'{prefix}_{i}')


def train(run_name, runs_path,
          base_model=config.BASE_MODEL,
          epochs=config.EPOCHS,
          imgsz=config.IMGSZ,
          batch=config.BATCH,
          lr0=config.LR0,
          patience=config.PATIENCE,
          optimizer=config.OPTIMIZER,
          seed=config.SEED,
          exist_ok=True,
          **extra):
    """
    Train a model and return the path of best.pt.

    `extra` forwards any additional argument to YOLO.train() — for example
    copy_paste=0.3 for the augmentation experiments.
    """
    print(f'\n  modelo    : {base_model}')
    print(f'  épocas    : {epochs}   imgsz: {imgsz}   batch: {batch}')
    print(f'  optimizer : {optimizer}   lr0: {lr0}')
    if extra:
        print(f'  extra     : {extra}')

    model = YOLO(base_model)
    model.train(
        data      = config.YAML_PATH,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        workers   = config.WORKERS,
        patience  = patience,
        optimizer = optimizer,
        lr0       = lr0,
        cos_lr    = config.COS_LR,
        seed      = seed,
        device    = config.DEVICE,
        project   = runs_path,
        name      = run_name,
        exist_ok  = exist_ok,
        verbose   = False,
        **extra
    )

    best = os.path.join(runs_path, run_name, 'weights', 'best.pt')
    print(f'\n  Mejor modelo: {best}')
    return best
