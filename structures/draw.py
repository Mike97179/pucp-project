"""
Drawing masks over the images.

Shared by the `benchmark` and `predecir` commands: this function used to be
copy-pasted in evaluar_benchmark.py and generar_mascaras.py.
"""

import cv2
import numpy as np
from ultralytics.utils.plotting import Colors

PALETTE = Colors()


def draw_masks(result, target, class_names, alpha=0.4):
    """
    Save the image with filled masks, boxes and labels.

    Returns how many objects were drawn. If the model detected nothing, the
    original image is saved unchanged.
    """
    img = result.orig_img.copy()

    if (result.masks is None or result.boxes is None
            or len(result.boxes) == 0):
        cv2.imwrite(target, img)
        return 0

    height, width = img.shape[:2]
    thickness = max(2, round((height + width) / 2 * 0.003))
    scale     = max(0.5, (height + width) / 2 * 0.0009)

    classes  = result.boxes.cls.tolist()
    confs    = result.boxes.conf.tolist()
    boxes    = result.boxes.xyxy.tolist()
    polygons = result.masks.xy

    # Highest to lowest confidence: the most certain ends up drawn on top
    order = sorted(range(len(classes)), key=lambda i: confs[i], reverse=True)

    overlay = img.copy()
    for i in order:
        pts = np.array(polygons[i], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], PALETTE(int(classes[i]), True))
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for i in order:
        cls_id = int(classes[i])
        color  = PALETTE(cls_id, True)
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f'{class_names[cls_id]} {confs[i]:.2f}'
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y_label = max(y1, th + 6)
        cv2.rectangle(img, (x1, y_label - th - baseline - 4),
                      (x1 + tw + 6, y_label), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1 + 3, y_label - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)

    cv2.imwrite(target, img)
    return len(order)
