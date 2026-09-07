"""
`validar` command — integrity check of dataset/ before training.

Ultralytics is tolerant with broken annotations: it skips what it cannot read
and trains anyway, so a corrupt label file shows up later as a class that
never learns. This command reads every .txt by hand and says exactly which
line is wrong.

Four checks: images without label, labels without image, polygons with fewer
than three points, and class_ids outside the range declared in data.yaml.
"""

import glob
import os

from .. import config, data

def _read_split_basenames():
    """Basenames already assigned to train/val/test."""
    names = set()
    for txt in (config.TRAIN_TXT, config.VAL_TXT, config.TEST_TXT):
        if os.path.isfile(txt):
            with open(txt) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        names.add(os.path.basename(line))
    return names


def _update_splits_if_needed(images):
    """Regenerate train/val/test.txt when images are missing from the splits."""
    benchmark_names = data.read_benchmark_names()
    split_names = _read_split_basenames()

    splittable = [img for img in images
                  if os.path.basename(img) not in benchmark_names]
    missing = [img for img in splittable
               if os.path.basename(img) not in split_names]

    if not missing:
        return

    print(f'\n  {len(missing)} imágenes nuevas detectadas fuera de los splits.')
    print('  Regenerando train.txt / val.txt / test.txt ...')
    data.generate_split(verbose=False)
    print('  Splits actualizados.')


# Un polígono necesita 3 puntos como mínimo, es decir 6 valores x,y detrás
# del class_id.
POLYGON_MIN_POINTS = 3

# Los listados largos se cortan: el detalle completo no cabe en pantalla.
MAX_LISTED = 15


def _basename(path):
    """File name without extension, which is what links image and label."""
    return os.path.splitext(os.path.basename(path))[0]


def _label_files(lbl_dir=config.LBL_DIR):
    """Every annotation .txt, sorted."""
    return sorted(glob.glob(os.path.join(lbl_dir, '*.txt')))


def _listing(items, indent='    '):
    """Print a capped list, saying how many were left out."""
    for item in items[:MAX_LISTED]:
        print(f'{indent}{item}')
    if len(items) > MAX_LISTED:
        print(f'{indent}... y {len(items) - MAX_LISTED} más')


def _check_line(parts, n_classes):
    """
    Validate one annotation line: `class_id x1 y1 x2 y2 ...` normalised.

    Returns (error, warning), each None when there is nothing to report. An
    error means the line is unusable; a warning means it is readable but
    suspicious.
    """
    try:
        class_id = int(parts[0])
    except ValueError:
        return f'class_id no numérico: "{parts[0]}"', None

    if not 0 <= class_id < n_classes:
        return (f'class_id {class_id} fuera de rango '
                f'(las clases van de 0 a {n_classes - 1})'), None

    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return 'hay coordenadas que no son números', None

    if len(coords) < POLYGON_MIN_POINTS * 2:
        return (f'polígono de {len(coords) // 2} puntos '
                f'({len(coords)} valores): hacen falta al menos '
                f'{POLYGON_MIN_POINTS}'), None

    if len(coords) % 2:
        return (f'{len(coords)} coordenadas: van en pares x,y, '
                f'sobra una'), None

    # Las coordenadas de YOLO están normalizadas: fuera de 0..1 el polígono
    # se sale de la imagen. Se avisa pero no invalida el archivo.
    outside = [c for c in coords if not 0.0 <= c <= 1.0]
    if outside:
        return None, (f'{len(outside)} coordenadas fuera de 0..1 '
                      f'(mín {min(outside):.3f}, máx {max(outside):.3f})')

    return None, None


def _check_label_file(path, n_classes):
    """
    Validate one .txt.

    Returns (errors, warnings, lines), each list holding 'línea N: motivo'.
    An empty file is valid: it means an image with no objects.
    """
    errors, warnings, lines = [], [], 0

    with open(path) as f:
        for number, raw in enumerate(f, start=1):
            parts = raw.split()
            if not parts:
                continue

            lines += 1
            error, warning = _check_line(parts, n_classes)
            if error:
                errors.append(f'línea {number}: {error}')
            if warning:
                warnings.append(f'línea {number}: {warning}')

    return errors, warnings, lines


def _orphans(images, labels):
    """
    Cross images and labels by file name.

    Returns (images without label, labels without image) as sorted names.
    """
    image_names = {_basename(i) for i in images}
    label_names = {_basename(l) for l in labels}

    return (sorted(image_names - label_names),
            sorted(label_names - image_names))


def run(args):
    class_names = config.load_class_names()
    images      = data.list_images()
    labels      = _label_files()

    print('\n' + '=' * 60)
    print(' VALIDACIÓN DEL DATASET')
    print('=' * 60)
    print(f'  Imágenes : {len(images)} en {config.IMG_DIR}')
    print(f'  Labels   : {len(labels)} en {config.LBL_DIR}')
    print(f'  Clases   : {len(class_names)} '
          f'(class_id válido: 0..{len(class_names) - 1})')

    if not images and not labels:
        print('\n  No hay imágenes ni labels: nada que validar.')
        print('  Coloca el dataset en dataset/images y dataset/labels.\n')
        return 1

    # ---------------- huérfanos en las dos direcciones ----------------
    without_label, without_image = _orphans(images, labels)

    print('\n' + '-' * 60)
    print(' EMPAREJAMIENTO IMAGEN <-> LABEL')
    print('-' * 60)

    if without_label:
        print(f'\n  Imágenes sin su .txt en {config.LBL_DIR}: '
              f'{len(without_label)}')
        _listing(without_label)
    if without_image:
        print(f'\n  Labels sin su imagen en {config.IMG_DIR}: '
              f'{len(without_image)}')
        _listing(without_image)
    if not without_label and not without_image:
        print('\n  Correcto: cada imagen tiene su label y al revés.')

    # ---------------- contenido de cada .txt ----------------
    files_with_errors   = {}
    files_with_warnings = {}
    empty_files         = []
    total_lines         = 0

    for path in labels:
        errors, warnings, lines = _check_label_file(path, len(class_names))
        total_lines += lines

        if not lines:
            empty_files.append(os.path.basename(path))
        if errors:
            files_with_errors[os.path.basename(path)] = errors
        if warnings:
            files_with_warnings[os.path.basename(path)] = warnings

    print('\n' + '-' * 60)
    print(' CONTENIDO DE LOS LABELS')
    print('-' * 60)

    if files_with_errors:
        print(f'\n  Archivos con errores: {len(files_with_errors)}')
        for name in sorted(files_with_errors)[:MAX_LISTED]:
            print(f'\n    {name}')
            _listing(files_with_errors[name], indent='      ')
        if len(files_with_errors) > MAX_LISTED:
            print(f'\n    ... y {len(files_with_errors) - MAX_LISTED} '
                  f'archivos más con errores')
    else:
        print(f'\n  Correcto: {total_lines} líneas revisadas, todas con '
              f'class_id válido y polígonos de {POLYGON_MIN_POINTS}+ puntos.')

    if files_with_warnings:
        print(f'\n  Archivos con avisos: {len(files_with_warnings)}')
        for name in sorted(files_with_warnings)[:MAX_LISTED]:
            print(f'    {name}: {files_with_warnings[name][0]}')
        if len(files_with_warnings) > MAX_LISTED:
            print(f'    ... y {len(files_with_warnings) - MAX_LISTED} más')

    if empty_files:
        print(f'\n  Archivos vacíos (imagen sin objetos, es válido): '
              f'{len(empty_files)}')
        _listing(empty_files)

    # ---------------- resumen ----------------
    print('\n' + '=' * 60)
    print(' RESUMEN')
    print('=' * 60)
    print(f'  Total de imágenes        : {len(images)}')
    print(f'  Total de labels          : {len(labels)}')
    print(f'  Imágenes sin label       : {len(without_label)}')
    print(f'  Labels sin imagen        : {len(without_image)}')
    print(f'  Labels con errores       : {len(files_with_errors)}')
    print(f'  Líneas revisadas         : {total_lines}')
    print('=' * 60)

    problems = len(without_label) + len(without_image) + len(files_with_errors)
    if problems:
        print(f'\n  {problems} problemas que corregir antes de entrenar.\n')
        return 1

    print('\n  Dataset válido: listo para entrenar.')

    # ---------------- actualizar splits si hay imágenes nuevas ----------------
    _update_splits_if_needed(images)

    print()
    return 0


def register(subparsers):
    p = subparsers.add_parser(
        'validate', help='Comprueba la integridad de imágenes y anotaciones')
    p.set_defaults(func=run)
    return p
