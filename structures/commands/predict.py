"""
`predecir` command — masks over loose images or a whole split.

Replaces generar_mascaras.py and probar_modelo.py: draws the predictions and
reports the object count per class and per image.

If --model is not given, the trained models are listed and the user is asked
which one to use: one is never picked by default behind their back.
"""

import os

import pandas as pd
from ultralytics import YOLO

from .. import config, data, draw, models


def _resolve_input(args):
    """Return the list of images to process, from --images or --split."""
    if args.images:
        paths = []
        for entry in args.images:
            if os.path.isdir(entry):
                paths.extend(data.list_images(entry))
            elif os.path.isfile(entry):
                paths.append(entry)
            else:
                print(f'  AVISO: no existe {entry}')
        return paths

    txt = {'train': config.TRAIN_TXT,
           'val'  : config.VAL_TXT,
           'test' : config.TEST_TXT}[args.split]

    if not os.path.isfile(txt):
        raise FileNotFoundError(
            f'No existe {txt}. Corre primero un comando que genere el split '
            f'(por ejemplo `entrenar`).')

    with open(txt) as f:
        # dict.fromkeys: drops duplicates if train.txt carries oversampling
        return list(dict.fromkeys(ln.strip() for ln in f if ln.strip()))


def run(args):
    class_names = config.load_class_names()
    weights = models.ask_one(
        args.model, title='¿Con qué modelo quieres predecir?')

    images = _resolve_input(args)
    if not images:
        print('No hay imágenes que procesar.')
        return

    output = os.path.join(config.BASE_PATH, args.output)
    os.makedirs(output, exist_ok=True)

    print(f'Modelo   : {weights}')
    print(f'Imágenes : {len(images)}')
    print(f'Salida   : {output}\n')

    model = YOLO(weights)
    rows = []

    for path in images:
        base = os.path.splitext(os.path.basename(path))[0]
        result = model.predict(path, conf=args.conf, verbose=False)[0]

        target = os.path.join(output, f'{base}_pred.jpg')
        total = draw.draw_masks(result, target, class_names, alpha=args.alpha)

        row = {'image': os.path.basename(path)}
        counts = {n: 0 for n in class_names}
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_id in result.boxes.cls.tolist():
                counts[class_names[int(cls_id)]] += 1
        row.update(counts)
        row['TOTAL'] = total
        rows.append(row)

        detail = ', '.join(f'{n}={c}' for n, c in counts.items() if c)
        print(f'  {os.path.basename(path):<25} [{total:>3} obj] {detail}')

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output, 'detection_counts.csv')
    df.to_csv(csv_path, index=False)

    print(f'\nConteo guardado en: {csv_path}')
    print('\nTotales por clase:')
    for name in class_names:
        total = int(df[name].sum())
        if total:
            print(f'  {name:<18} {total:>5}')
    print(f'  {"TOTAL":<18} {int(df["TOTAL"].sum()):>5}')


def register(subparsers):
    p = subparsers.add_parser(
        'predict', help='Dibuja máscaras y cuenta objetos detectados')
    p.add_argument('--model', dest='model', default=None,
                   help='Nombre del registro o ruta a un .pt '
                        '(si se omite, se pregunta listando los disponibles)')
    p.add_argument('--images', dest='images', nargs='+', default=None,
                   help='Imágenes o carpetas concretas (default: usa --split)')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                   help='Partición a procesar si no se pasa --images')
    p.add_argument('--output', dest='output', default='predicciones',
                   help='Carpeta de salida (default: predicciones/)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--alpha', type=float, default=0.4)
    p.set_defaults(func=run)
    return p
