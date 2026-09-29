"""
`evaluate` command — evaluates a model on a given dataset.

Takes a .pt weights path and a .txt file with image paths (like the test.txt
saved in each model's folder).  Temporarily swaps the active test.txt so
model.val() runs on the provided images, then restores the original.
"""

import os
import shutil
import sys

from .. import config, evaluate


def run(args):
    weights = os.path.abspath(args.model)
    dataset = os.path.abspath(args.dataset)

    if not os.path.isfile(weights):
        sys.exit(f'No se encontró el modelo: {weights}')

    if not weights.endswith('.pt'):
        sys.exit(f'El modelo debe ser un archivo .pt: {weights}')

    if os.path.isdir(dataset):
        candidate = os.path.join(dataset, 'test.txt')
        if os.path.isfile(candidate):
            dataset = candidate
        else:
            sys.exit(f'No se encontró test.txt en {dataset}')

    if not os.path.isfile(dataset):
        sys.exit(f'No se encontró el dataset: {dataset}')

    class_names = config.load_class_names()

    print(f'  Modelo  : {weights}')
    print(f'  Dataset : {dataset}')

    with open(dataset) as f:
        n_images = sum(1 for line in f if line.strip())
    print(f'  Imágenes: {n_images}')

    backup = None
    need_swap = os.path.abspath(dataset) != os.path.abspath(config.TEST_TXT)

    if need_swap:
        if os.path.isfile(config.TEST_TXT):
            backup = config.TEST_TXT + '.bak'
            shutil.copy2(config.TEST_TXT, backup)
        shutil.copy2(dataset, config.TEST_TXT)

    try:
        print('\nEjecutando model.val()...\n')
        metrics = evaluate.validate(weights, split='test', verbose=True)

        print('\n' + '=' * 70)
        print(' MÉTRICAS POR CLASE (segmentación)')
        print('=' * 70)
        df = evaluate.per_class_metrics(metrics, class_names)
        print(df.to_string(index=False))

        totals = evaluate.summary(metrics)
        print(f'\n  box_mAP50  : {totals["mAP50_detection"]}')
        print(f'  mask_mAP50 : {totals["mAP50_segmentation"]}')
        print(f'  box_mAP50-95  : {totals["mAP5095_detection"]}')
        print(f'  mask_mAP50-95 : {totals["mAP5095_segmentation"]}')

        evaluate.free_memory()
    finally:
        if need_swap:
            if backup and os.path.isfile(backup):
                shutil.move(backup, config.TEST_TXT)
            elif not backup:
                os.remove(config.TEST_TXT)


def register(subparsers):
    p = subparsers.add_parser(
        'evaluate',
        help='Evalúa un modelo sobre un dataset específico')
    p.add_argument('model',
                   help='Ruta al archivo .pt del modelo')
    p.add_argument('dataset',
                   help='Ruta al .txt con las imágenes a evaluar '
                        '(o carpeta que contenga test.txt)')
    p.set_defaults(func=run)
    return p
