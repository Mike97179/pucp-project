"""
Evaluation — global metrics, per-class metrics and confusion matrix.

Every figure comes from model.val() over a concrete split, not from the last
row of results.csv (which is the last epoch, not the best one).
"""

import gc

import pandas as pd
import torch
from ultralytics import YOLO

from . import config


def validate(weights, split='test', verbose=False):
    """Run model.val() and return the Ultralytics metrics object."""
    model = YOLO(weights)
    return model.val(data=config.YAML_PATH, split=split, verbose=verbose)


def per_class_metrics(metrics, class_names, task='seg'):
    """
    DataFrame with precision / recall / mAP50 / mAP50-95 per class.

    `ap_class_index` gives the real indices of the evaluated classes; without
    it the order does not line up with `class_names` when some class has no
    instances. Classes without instances are reported as 0 so the table always
    has one row per class.
    """
    result   = getattr(metrics, task)
    evaluated = list(result.ap_class_index)

    rows = []
    for pos, class_idx in enumerate(evaluated):
        p, r, ap50, ap = result.class_result(pos)
        rows.append({
            'class'    : class_names[class_idx],
            'precision': round(p, 4),
            'recall'   : round(r, 4),
            'mAP50'    : round(ap50, 4),
            'mAP50-95' : round(ap, 4),
        })

    without_data = [class_names[i] for i in range(len(class_names))
                    if i not in evaluated]
    for name in without_data:
        rows.append({'class': name, 'precision': 0.0, 'recall': 0.0,
                     'mAP50': 0.0, 'mAP50-95': 0.0})

    if without_data:
        print(f'\n  AVISO: sin instancias en el set evaluado -> {without_data}')

    return pd.DataFrame(rows)


def summary(metrics):
    """Global detection and segmentation mAP."""
    return {
        'mAP50_detection'     : round(metrics.box.map50, 4),
        'mAP50_segmentation'  : round(metrics.seg.map50, 4),
        'mAP5095_detection'   : round(metrics.box.map, 4),
        'mAP5095_segmentation': round(metrics.seg.map, 4),
    }


def confusion_matrix(metrics):
    """Confusion matrix as integers, with the extra background row/column."""
    return metrics.confusion_matrix.matrix.astype(int)


def free_memory():
    """Release GPU memory between models when evaluating in batch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
