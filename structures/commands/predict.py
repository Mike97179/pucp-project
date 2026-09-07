"""
`predict` command — masks over a split, with cross-model comparison.

Each model's output goes to predicciones/<model_name>/: masked images and a
detection_counts.csv.  A summary in predicciones/ compares predicted counts
against the ground truth annotations for every image and class, across all
models that have been used.

When a model is archived from the leaderboard its predictions are removed and
the summary is regenerated without it.
"""

import os
import shutil

import pandas as pd
from ultralytics import YOLO

from .. import config, data, draw, models, plots


PREDICT_DIR = os.path.join(config.BASE_PATH, 'predicciones')


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
            f'(por ejemplo `train`).')

    with open(txt) as f:
        return list(dict.fromkeys(ln.strip() for ln in f if ln.strip()))


def _find_model_name(weights_path):
    """Model name from the registry, or filename without .pt."""
    try:
        df = models.registry()
        match = df[df['path'] == weights_path]
        if not match.empty:
            return match.iloc[0]['name']
    except FileNotFoundError:
        pass
    return os.path.splitext(os.path.basename(weights_path))[0]


def _ground_truth_counts(img_name, class_names):
    """Ground truth instance count per class for one image."""
    img_path = os.path.join(config.IMG_DIR, img_name)
    gt = data.read_ground_truth(img_path)
    return {name: gt.get(i, 0) for i, name in enumerate(class_names)}


# ------------------------------------------------------------------
# Summary: cross-model comparison against ground truth
# ------------------------------------------------------------------
def rebuild_predict_summary(class_names=None, predict_dir=None):
    """
    Rebuild predicciones/summary.csv from every model subdirectory.

    Reads each model's detection_counts.csv and compares the predicted counts
    against the ground truth labels.  The result is a MultiIndex CSV:

        (model, class, real/pred/diff)

    indexed by image name.  Also generates a comparison chart.
    """
    if predict_dir is None:
        predict_dir = PREDICT_DIR
    if class_names is None:
        class_names = config.load_class_names()

    if not os.path.isdir(predict_dir):
        return None

    model_data = {}
    for entry in sorted(os.listdir(predict_dir)):
        det_csv = os.path.join(predict_dir, entry, 'detection_counts.csv')
        if os.path.isfile(det_csv):
            model_data[entry] = pd.read_csv(det_csv)

    summary_csv = os.path.join(predict_dir, 'summary.csv')
    summary_png = os.path.join(predict_dir, 'summary.png')

    if not model_data:
        for path in (summary_csv, summary_png):
            if os.path.isfile(path):
                os.remove(path)
        return None

    all_images = sorted(set().union(
        *(set(df['image']) for df in model_data.values())
    ))

    tuples = []
    for model_name in model_data:
        for cls in class_names:
            tuples.extend([
                (model_name, cls, 'real'),
                (model_name, cls, 'pred'),
                (model_name, cls, 'diff'),
            ])

    columns = pd.MultiIndex.from_tuples(
        tuples, names=['modelo', 'clase', 'métrica'])
    summary = pd.DataFrame(0, index=all_images, columns=columns, dtype=int)
    summary.index.name = 'imagen'

    for img_name in all_images:
        gt = _ground_truth_counts(img_name, class_names)

        for model_name, det_df in model_data.items():
            img_row = det_df[det_df['image'] == img_name]
            for cls in class_names:
                real = gt.get(cls, 0)
                pred = (int(img_row[cls].iloc[0])
                        if not img_row.empty and cls in img_row.columns
                        else 0)
                summary.at[img_name, (model_name, cls, 'real')] = real
                summary.at[img_name, (model_name, cls, 'pred')] = pred
                summary.at[img_name, (model_name, cls, 'diff')] = pred - real

    summary.to_csv(summary_csv)
    print(f'  Resumen comparativo: {summary_csv}')

    gt_totals = [
        summary[(list(model_data)[0], cls, 'real')].sum()
        for cls in class_names
    ]
    pred_per_model = {
        mn: [summary[(mn, cls, 'pred')].sum() for cls in class_names]
        for mn in model_data
    }
    plots.predict_summary(
        class_names, gt_totals, pred_per_model, summary_png)

    return summary


def clean_model_predictions(model_name, predict_dir=None):
    """
    Remove a model's predictions and rebuild the summary without it.

    Called by models.archive_model() when a model leaves the leaderboard.
    """
    if predict_dir is None:
        predict_dir = PREDICT_DIR

    mdir = os.path.join(predict_dir, model_name)
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)
        print(f'  Limpieza: predicciones de "{model_name}" eliminadas')
        rebuild_predict_summary(predict_dir=predict_dir)


# ------------------------------------------------------------------
# Main command
# ------------------------------------------------------------------
def run(args):
    class_names = config.load_class_names()

    if args.model:
        weights    = models.resolve(args.model)
        model_name = _find_model_name(weights)
    else:
        selection  = models.ask(
            multiple=False, allow_all=False,
            title='¿Con qué modelo quieres predecir?')
        model_name = selection[0]['name']
        weights    = selection[0]['path']

    images = _resolve_input(args)
    if not images:
        print('No hay imágenes que procesar.')
        return

    predict_dir = os.path.join(config.BASE_PATH, args.output)
    output      = os.path.join(predict_dir, model_name)
    os.makedirs(output, exist_ok=True)

    print(f'Modelo   : {model_name} ({weights})')
    print(f'Imágenes : {len(images)}')
    print(f'Salida   : {output}\n')

    model = YOLO(weights)
    rows  = []

    for path in images:
        base   = os.path.splitext(os.path.basename(path))[0]
        result = model.predict(path, conf=args.conf, verbose=False)[0]

        target = os.path.join(output, f'{base}_pred.jpg')
        total  = draw.draw_masks(result, target, class_names, alpha=args.alpha)

        row    = {'image': os.path.basename(path)}
        counts = {n: 0 for n in class_names}
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_id in result.boxes.cls.tolist():
                counts[class_names[int(cls_id)]] += 1
        row.update(counts)
        row['TOTAL'] = total
        rows.append(row)

        detail = ', '.join(f'{n}={c}' for n, c in counts.items() if c)
        print(f'  {os.path.basename(path):<25} [{total:>3} obj] {detail}')

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(output, 'detection_counts.csv')
    df.to_csv(csv_path, index=False)

    print(f'\nConteo guardado en: {csv_path}')
    print('\nTotales por clase:')
    for name in class_names:
        total = int(df[name].sum())
        if total:
            print(f'  {name:<18} {total:>5}')
    print(f'  {"TOTAL":<18} {int(df["TOTAL"].sum()):>5}')

    # Cross-model summary
    print('\nActualizando resumen comparativo...')
    rebuild_predict_summary(class_names, predict_dir)


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
