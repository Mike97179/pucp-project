"""
`stats` command — what is actually inside the dataset.

Read-only inspection of dataset/: no model is loaded and nothing is trained,
so it runs in seconds. Answers the questions that come up before blaming the
model for a bad mAP50: how many instances each class really has, how the
images ended up split, and which ones carry no annotation at all.

A class with a handful of instances is not a training problem, it is a data
problem: that is what the warning at the end is about.
"""

import os

import cv2
import numpy as np
import pandas as pd

from .. import config, data


# Debajo de este número de instancias una clase casi nunca llega a aprenderse:
# es el umbral que separa "entrena mal" de "no hay datos".
MIN_INSTANCES = 50

# Las imágenes sin anotar se listan, pero sin llenar la pantalla.
MAX_LISTED = 20

SPLITS = [('train', config.TRAIN_TXT),
          ('val',   config.VAL_TXT),
          ('test',  config.TEST_TXT)]


def _count(amount, singular, plural):
    """'1 imagen' / '3 imágenes': los avisos se leen mal en singular."""
    return f'{amount} {singular if amount == 1 else plural}'


def _read_split(txt_path):
    """Non-empty lines of a split file, or None if the file is not there."""
    if not os.path.isfile(txt_path):
        return None
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def _class_table(images, class_names):
    """
    Instances per class over the whole dataset, most frequent first.

    Returns the DataFrame so the caller can reuse the counts for the warning
    instead of walking the labels twice.
    """
    instances = data.count_instances(images, len(class_names))
    total     = sum(instances)

    df = pd.DataFrame({
        'clase'     : class_names,
        'instancias': instances,
    })
    df['%'] = (df['instancias'] / total * 100).round(1) if total else 0.0
    df = df.sort_values('instancias', ascending=False).reset_index(drop=True)

    print('\n' + '=' * 60)
    print(' INSTANCIAS POR CLASE (dataset completo)')
    print('=' * 60)
    print(df.to_string(index=False))
    print(f'\n  Total: {total} instancias en {len(images)} imágenes')

    return df


def _split_images(split_txt):
    """Unique image paths from a split file, resolved to full dataset paths."""
    lines = _read_split(split_txt)
    if lines is None:
        return []
    seen = set()
    paths = []
    for line in lines:
        base = os.path.basename(line)
        if base not in seen:
            seen.add(base)
            paths.append(os.path.join(config.IMG_DIR, base))
    return paths


def _split_distribution(images):
    """Images per split, warning when the split files are missing or stale."""
    print('\n' + '=' * 60)
    print(f' DISTRIBUCIÓN TRAIN / VAL / TEST (semilla: {config.SEED})')
    print('=' * 60)

    missing = [name for name, txt in SPLITS if _read_split(txt) is None]
    if missing:
        print(f'\n  No existen los .txt del split ({", ".join(missing)}).')
        print('  Genéralos con `python pucp_segmentation.py split`.')
        return

    rows, in_splits = [], set()
    for name, txt in SPLITS:
        lines  = _read_split(txt)
        unique = {os.path.basename(line) for line in lines}
        in_splits |= unique
        rows.append({
            'split'    : name,
            'imágenes' : len(unique),
            'líneas'   : len(lines),
        })

    df = pd.DataFrame(rows)
    total = df['imágenes'].sum()
    df['%'] = (df['imágenes'] / total * 100).round(1) if total else 0.0
    print(df.to_string(index=False))

    # train.txt repite líneas cuando se entrenó con oversampling: si quedaron
    # repetidas, el próximo entrenamiento las usaría sin avisar.
    repeated = df[df['líneas'] > df['imágenes']]
    if not repeated.empty:
        for _, row in repeated.iterrows():
            print(f'\n  AVISO: {row["split"]}.txt tiene '
                  f'{_count(row["líneas"], "línea", "líneas")} para '
                  f'{_count(row["imágenes"], "imagen", "imágenes")}: hay '
                  f'repeticiones de oversampling.')
        print('  Quítalas con `python pucp_segmentation.py split --restore`.')

    benchmark = data.read_benchmark_names()
    if benchmark:
        print(f'\n  Reservadas para benchmark (fuera del split): '
              f'{_count(len(benchmark), "imagen", "imágenes")}')

    # Lo que hay en dataset/images pero en ningún .txt: normalmente son las de
    # benchmark, y si no, es que el split se quedó viejo.
    loose = {os.path.basename(i) for i in images} - in_splits - benchmark
    if loose:
        print(f'\n  AVISO: {_count(len(loose), "imagen", "imágenes")} no '
              f'{"está" if len(loose) == 1 else "están"} en ningún split ni '
              f'en benchmark.txt.')
        print('  El split se generó antes de añadirlas: '
              'regenéralo con `split`.')


def show_class_per_split(class_names, from_split=False):
    """Per-class instance counts broken down by split + benchmark."""
    n = len(class_names)
    all_splits = SPLITS + [('benchmark', config.BENCHMARK_TXT)]

    missing = [name for name, txt in SPLITS if _read_split(txt) is None]
    if missing:
        return

    split_counts = {}
    for name, txt in all_splits:
        imgs = _split_images(txt)
        split_counts[name] = data.count_instances(imgs, n)

    print('\n' + '=' * 60)
    print(' INSTANCIAS POR CLASE Y SPLIT')
    print('=' * 60)

    header = f'{"clase":<18}'
    for name, _ in all_splits:
        header += f'{name:>10}'
    header += f'{"total":>10}'
    print(header)
    print('-' * len(header))

    warnings = []
    for i, cls in enumerate(class_names):
        row = f'{cls:<18}'
        cls_total = 0
        for name, _ in all_splits:
            c = split_counts[name][i]
            cls_total += c
            row += f'{c:>10}'
        row += f'{cls_total:>10}'
        print(row)

        test_count = split_counts['test'][i]
        if cls_total > 0 and test_count == 0:
            warnings.append(cls)

    if warnings:
        print(f'\n  AVISO: {", ".join(warnings)} no '
              f'{"tiene" if len(warnings) == 1 else "tienen"} instancias en '
              f'test.')
        print('  El modelo no se puede evaluar en esa clase con este split.')
        if from_split:
            print('  Prueba con otra semilla: '
                  '`python pucp_segmentation.py split --seed N`.')
        else:
            print('  Regenera el split con '
                  '`python pucp_segmentation.py split`.')


def _unannotated(images):
    """List the images with no annotation, split file-missing vs file-empty."""
    print('\n' + '=' * 60)
    print(' IMÁGENES SIN ANOTACIÓN')
    print('=' * 60)

    no_file, empty_file = [], []
    for image in images:
        if not os.path.isfile(data.label_path(image)):
            no_file.append(image)
        elif not data.read_ground_truth(image):
            empty_file.append(image)

    def report(title, items):
        if not items:
            return
        print(f'\n  {title}: {len(items)}')
        for path in items[:MAX_LISTED]:
            print(f'    {os.path.basename(path)}')
        if len(items) > MAX_LISTED:
            print(f'    ... y {len(items) - MAX_LISTED} más')

    report(f'Sin archivo .txt en {config.LBL_DIR}', no_file)
    report('Con .txt vacío (imagen de fondo, sin objetos)', empty_file)

    if not no_file and not empty_file:
        print('\n  Ninguna: todas las imágenes tienen anotaciones.')
        return

    if no_file:
        print('\n  Las imágenes sin .txt entran igual al split y el modelo las '
              'aprende')
        print('  como fondo. Anótalas o sácalas de dataset/images.')


def _weak_classes(df):
    """Warn about the classes that do not have enough instances to be learnt."""
    empty = df[df['instancias'] == 0]
    weak  = df[(df['instancias'] > 0) & (df['instancias'] < MIN_INSTANCES)]

    if empty.empty and weak.empty:
        print('\n' + '=' * 60)
        print(f' Todas las clases superan las {MIN_INSTANCES} instancias.')
        print('=' * 60)
        return

    print('\n' + '=' * 60)
    print(' CLASES CON POCOS DATOS')
    print('=' * 60)

    if not empty.empty:
        print('\n  Sin ninguna instancia anotada '
              '(el modelo no las detectará nunca):')
        for _, row in empty.iterrows():
            print(f'    {row["clase"]}')

    if not weak.empty:
        print(f'\n  Menos de {MIN_INSTANCES} instancias '
              f'(muy probablemente ~0.0 de mAP50):')
        for _, row in weak.iterrows():
            print(f'    {row["clase"]:<20} '
                  f'{_count(row["instancias"], "instancia", "instancias")}')

        print('\n  Con tan pocos ejemplos el modelo casi nunca acierta esa '
              'clase, y')
        print('  el mAP50 global baja aunque el resto vaya bien. Anota más '
              'imágenes')
        print('  de esas clases, o entrena con --oversample para repetirlas.')


def _instances_per_image_excel(images, class_names):
    """Write an Excel with per-image instance counts, saved in dataset/."""
    n_classes = len(class_names)
    rows = []
    for img in images:
        gt = data.read_ground_truth(img)
        row = {'imagen': os.path.basename(img)}
        for i, name in enumerate(class_names):
            row[name] = gt.get(i, 0)
        row['total'] = sum(gt.values())
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('imagen').reset_index(drop=True)

    totals = {'imagen': 'TOTAL'}
    for name in class_names:
        totals[name] = df[name].sum()
    totals['total'] = df['total'].sum()
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    out_path = os.path.join(config.DATASET_PATH, 'instancias_por_imagen.xlsx')
    df.to_excel(out_path, index=False)

    print('\n' + '=' * 60)
    print(' CONTEO DE INSTANCIAS POR IMAGEN')
    print('=' * 60)
    print(f'  Guardado en: {out_path}')
    print(f'  {len(images)} imágenes, {len(class_names)} clases')


def _draw_ground_truth(images, class_names, alpha=0.4):
    """Draw segmentation masks from labels onto each image."""
    output_dir = os.path.join(config.DATASET_PATH, 'ground_truth')
    os.makedirs(output_dir, exist_ok=True)

    colors = config.CLASS_COLORS_BGR

    print('\n' + '=' * 60)
    print(' VISUALIZACIÓN GROUND TRUTH')
    print('=' * 60)

    drawn = 0
    for img_path in images:
        lbl_path = data.label_path(img_path)
        if not os.path.isfile(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        thickness = max(1, round((h + w) / 2 * 0.0015))
        scale = max(0.3, (h + w) / 2 * 0.0005)

        overlay = img.copy()
        annotations = []

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                pts = np.array(
                    [[int(x * w), int(y * h)] for x, y in zip(xs, ys)],
                    dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], colors[cls_id])
                    annotations.append((cls_id, pts))

        if not annotations:
            continue

        result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        for cls_id, pts in annotations:
            color = colors[cls_id]
            cv2.polylines(result, [pts], True, color, thickness, cv2.LINE_AA)

            x_min, y_min = pts.min(axis=0)
            label = class_names[cls_id]
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            y_label = max(y_min, th + 6)
            cv2.rectangle(result, (x_min, y_label - th - baseline - 4),
                          (x_min + tw + 6, y_label), color, -1, cv2.LINE_AA)
            cv2.putText(result, label, (x_min + 3, y_label - baseline - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

        base = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, base), result)
        drawn += 1

    print(f'  Carpeta: {output_dir}')
    print(f'  Imágenes generadas: {drawn}')


def run(args):
    class_names = config.load_class_names()
    images      = data.list_images()

    print('\n' + '=' * 60)
    print(' ESTADÍSTICAS DEL DATASET')
    print('=' * 60)
    print(f'  Imágenes  : {len(images)} en {config.IMG_DIR}')
    print(f'  Etiquetas : {config.LBL_DIR}')
    print(f'  Clases    : {len(class_names)} ({", ".join(class_names)})')

    if not images:
        print('\n  No hay ninguna imagen en dataset/images: no hay nada que '
              'contar.')
        return

    df = _class_table(images, class_names)
    _split_distribution(images)
    _unannotated(images)
    _weak_classes(df)
    _instances_per_image_excel(images, class_names)
    _draw_ground_truth(images, class_names)
    print()


def register(subparsers):
    p = subparsers.add_parser(
        'stats', help='Estadísticas del dataset: clases, splits y anotaciones')
    p.set_defaults(func=run)
    return p
