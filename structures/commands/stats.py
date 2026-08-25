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


def _split_distribution(images):
    """Images per split, warning when the split files are missing or stale."""
    print('\n' + '=' * 60)
    print(' DISTRIBUCIÓN TRAIN / VAL / TEST')
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
    print()


def register(subparsers):
    p = subparsers.add_parser(
        'stats', help='Estadísticas del dataset: clases, splits y anotaciones')
    p.set_defaults(func=run)
    return p
